# app.py
import os
import io
import time
import base64

import cv2
import torch
import numpy as np
import pydicom
import sqlite3
import segmentation_models_pytorch as smp
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from peft import PeftModel

from fastapi.middleware.cors import CORSMiddleware

# --- CONFIG ---
CKPT_PATH = "runs_unet/unet_resnet34_best.pth"  
IMG_SIZE  = 256
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DB_PATH = "inference_logs.db"
MODEL_VERSION = "unet_resnet34_v1" 

LLM_BASE_MODEL = "google/flan-t5-base"
LLM_ADAPTER_DIR = "outputs_lora_reporter"

def init_db():
    """Create SQLite DB and table if not exists."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS inference_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            filename TEXT,
            latency_ms INTEGER,
            model_version TEXT
        );
        """
    )
    conn.commit()
    conn.close()

class ReportRequest(BaseModel):
    level: str
    side: str
    muscle_area_mm2: float
    fat_infiltration_pct: float
    degeneration_grade: str

# --- LOAD MODEL ---
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=1,
    classes=1
).to(DEVICE)

state = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

init_db()

llm_tokenizer = AutoTokenizer.from_pretrained(LLM_BASE_MODEL)
llm_base = AutoModelForSeq2SeqLM.from_pretrained(LLM_BASE_MODEL)
llm_model = PeftModel.from_pretrained(llm_base, LLM_ADAPTER_DIR)
llm_model.to(DEVICE)
llm_model.eval()

app = FastAPI(title="Multifidus Segmentation API")


# --- HELPERS ---

def read_any_mri_bytes(filename: str, file_bytes: bytes) -> np.ndarray:
    """
    Read a DICOM (.dcm) or image (.png/.jpg/...) from raw bytes.
    Return 2D uint8 grayscale image.
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".dcm":
        f = io.BytesIO(file_bytes)
        dcm = pydicom.dcmread(f)
        img = dcm.pixel_array.astype(np.float32)
        img -= img.min()
        if img.max() > 0:
            img /= img.max()
        img = (img * 255).astype(np.uint8)
    else:
        # Assume standard image format
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Cannot read file as image.")
    return img


def run_inference_on_slice(img_uint8: np.ndarray) -> np.ndarray:
    """
    img_uint8: 2D uint8 image (H,W)
    Returns overlay image (H,W,3) in BGR.
    """
    h, w = img_uint8.shape

    # Resize & normalize for model
    img_resized = cv2.resize(img_uint8, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    x = torch.from_numpy(img_resized)[None, None, ...].to(DEVICE)  # (1,1,H,W)

    with torch.no_grad():
        pred = torch.sigmoid(model(x))[0, 0].cpu().numpy()

    mask = (pred > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Build overlay
    base = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
    color_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    overlay_bgr = cv2.addWeighted(base, 0.8, color_mask, 0.4, 0)

    return overlay_bgr


def bgr_image_to_base64_png(img_bgr: np.ndarray) -> str:
    """
    Encode BGR image as PNG and return base64 string.
    """
    success, buf = cv2.imencode(".png", img_bgr)
    if not success:
        raise RuntimeError("Failed to encode image.")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return b64

def build_llm_input_text(level, side, muscle_area_mm2, fat_infiltration_pct, degeneration_grade):
    return (
        "You are an AI assistant that generates short clinical-style reports for lumbar multifidus muscle degeneration.\n"
        "Given the following measurements, produce a report with exactly three sections: Summary, Risk, and Recommendation.\n"
        "Input measurements:\n"
        f"level: {level}\n"
        f"side: {side}\n"
        f"muscle_area_mm2: {muscle_area_mm2}\n"
        f"fat_infiltration_pct: {fat_infiltration_pct}\n"
        f"degeneration_grade: {degeneration_grade}"
    )


def generate_multifidus_report(level, side, muscle_area_mm2, fat_infiltration_pct, degeneration_grade):
    prompt = build_llm_input_text(level, side, muscle_area_mm2, fat_infiltration_pct, degeneration_grade)
    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True).to(DEVICE)

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_length=160,
            num_beams=4,
            do_sample=False 
        )

    text = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


# --- ROUTES ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    """
    Accepts .dcm or normal image, returns:
    - overlay image as base64 PNG
    - original width/height
    - inference latency (ms)
    """
    start_time = time.time()

    file_bytes = await file.read()
    try:
        img_uint8 = read_any_mri_bytes(file.filename, file_bytes)
        overlay_bgr = run_inference_on_slice(img_uint8)
        overlay_b64 = bgr_image_to_base64_png(overlay_bgr)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )

    latency_ms = int((time.time() - start_time) * 1000)

    # --- SQLite logging ---
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO inference_logs (filename, latency_ms, model_version)
            VALUES (?, ?, ?)
            """,
            (file.filename, latency_ms, MODEL_VERSION),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        # Fail silently for logging errors, but print for debugging
        print(f"[logging error] {e}")

    # Console log (still useful)
    print(f"[segment] filename={file.filename} latency_ms={latency_ms}")

    return {
        "filename": file.filename,
        "latency_ms": latency_ms,
        "overlay_base64": overlay_b64,
        "width": overlay_bgr.shape[1],
        "height": overlay_bgr.shape[0]
    }

@app.post("/report")
async def report(req: ReportRequest):
    """
    Generate a clinical-style multifidus report from numeric measurements.
    """
    try:
        text = generate_multifidus_report(
            level=req.level,
            side=req.side,
            muscle_area_mm2=req.muscle_area_mm2,
            fat_infiltration_pct=req.fat_infiltration_pct,
            degeneration_grade=req.degeneration_grade,
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    return {
        "level": req.level,
        "side": req.side,
        "muscle_area_mm2": req.muscle_area_mm2,
        "fat_infiltration_pct": req.fat_infiltration_pct,
        "degeneration_grade": req.degeneration_grade,
        "report": text,
    }
