
# Multifidus MRI Segmentation & Reporting Demo

End-to-end AI service that segments the lumbar multifidus muscle on axial T2 MRI slices and exposes the model through a FastAPI REST API.  
The project also includes a small LoRA-fine-tuned language model that generates short clinical-style summaries from numeric measurements.

This project demonstrates **full-stack AI engineering capability**:
- CV model training + preprocessing  
- API deployment (Docker + FastAPI)  
- LLM fine-tuning (LoRA)  
- Frontend web demo (TailwindCSS + static hosting)  
- Production-style logging + architecture  

---

## 🚀 Live Demo

- **Frontend (Netlify):** https://multifidus-demo.netlify.app  
- **Backend API (Render, Swagger UI):** https://multifidus-api.onrender.com/docs  

---

## 📌 What This Project Does

- Accepts a single axial T2-weighted lumbar MRI slice (`.dcm` or `.png/.jpg`).
- Runs a U-Net (ResNet-34 encoder) segmentation model to highlight the multifidus muscle.
- Serves results via a Dockerized FastAPI backend deployed on Render.
- Optionally generates a short “clinical-style” summary using a LoRA-fine-tuned Flan-T5 model.

---

### Components

* **Frontend:** Static TailwindCSS site on Netlify.
* **Backend:** FastAPI + Uvicorn inside a Docker container.
* **Segmentation model:** U-Net with ResNet-34 encoder.
* **LLM module:** Flan-T5 with LoRA adapters (optional endpoint).
* **Logging:** SQLite (filename, latency, model version).

---

## 🔄 Pipeline Details

### 🧠 Segmentation Pipeline

1. **Input formats**

   * `.dcm` via `pydicom`
   * `.png` / `.jpg` grayscale via OpenCV

2. **Preprocessing**

   * Convert to float32
   * Min–max normalize to `[0,1]`
   * Resize to `256×256`
   * Tensor shape `(1,1,H,W)`

3. **Model**

   * U-Net with ResNet-34 encoder
   * 1 output class → sigmoid → threshold at 0.5

4. **Postprocessing**

   * Resize mask to original shape
   * Apply JET colormap
   * Alpha-blend overlay
   * Encode as PNG, return as Base64

---

### 📝 LoRA Reporting Pipeline (Optional)

1. **Input JSON fields**

   * `level` (`"L4-L5"`, `"L5-S1"`, etc.)
   * `side` (`"left"`, `"right"`, `"bilateral"`)
   * `muscle_area_mm2`
   * `fat_infiltration_pct`
   * `degeneration_grade`

2. **Model**

   * `google/flan-t5-base`
   * LoRA fine-tuning on attention projections (`q`, `v`)
   * Trainable parameters: **~0.36%** of the base model

3. **Output format**

   * `Summary: …`
   * `Risk: …`
   * `Recommendation: …`

---

## 🧩 Models & Data

### 🔹 Segmentation Model (U-Net ResNet-34)

* **Parameters:** ~24.8M
* **Training data:** Axial T2 MRI slices with ITK-Snap manual MF segmentation
* **Training style:**

  * Slice-wise
  * 256×256
  * Horizontal flips, light rotation
* **Loss:** BCE with sigmoid

---

### 🔹 LoRA LLM Model (Flan-T5)

* **Base:** `google/flan-t5-base` (~248M params)
* **Fine-tuning method:** LoRA (`r=8`, dropout 0.1)
* **Dataset:** ~20 handcrafted examples mapping metrics → templated summaries

---

## 📡 API Endpoints

### ➤ `POST /segment`

Run segmentation on uploaded MRI slice.

#### Request

```
Content-Type: multipart/form-data
file: <MRI slice>
```

#### Response

```json
{
  "filename": "slice01.dcm",
  "latency_ms": 88,
  "overlay_base64": "<base64PNG>",
  "width": 512,
  "height": 512
}
```

---

### ➤ `POST /report`  *(Optional)*

Generate a clinical-style text summary.

#### Request

```json
{
  "level": "L4-L5",
  "side": "left",
  "muscle_area_mm2": 423,
  "fat_infiltration_pct": 18,
  "degeneration_grade": "mild"
}
```

#### Response

```json
{
  "report": "Summary: ...\nRisk: ...\nRecommendation: ..."
}
```

Interactive API docs: `/docs`

---

## 🎨 Frontend Demo (TailwindCSS)

Features:

* Drag-and-drop or file-browse MRI upload
* Clean medical/demo-focused UI
* Real-time call to backend `/segment`
* Displays inference latency
* Error handling + loading state
* Optional LLM summary panel

Deployed at: **[https://multifidus-demo.netlify.app](https://multifidus-demo.netlify.app)**

---

## 🛠️ Local Development

### 1) Clone

```bash
git clone https://github.com/<your-username>/multifidus-api.git
cd multifidus-api
```

### 2) Install

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3) Run

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open: `http://localhost:8000/docs`

---

## 🐳 Docker Deployment

### Build

```bash
docker build -t multifidus-api .
```

### Run

```bash
docker run -p 8000:8000 multifidus-api
```

Backend now live at: `http://localhost:8000`

---

## ☁️ Cloud Deployment

### Backend (Render)

* Dockerized FastAPI service
* Detects port 8000 automatically
* Production process:

  ```
  uvicorn app:app --host 0.0.0.0 --port 8000
  ```

### Frontend (Netlify)

* Static `index.html` + TailwindCSS
* Calls the deployed backend URL

---

## ⚠️ Limitations & Future Work

### Limitations

* Not a clinical model
* Single-slice only (no 3D context)
* No ONNX/TensorRT optimization yet

### Future improvements

* Volume-level analysis + spinal level auto-detection
* Real annotated datasets & multi-center evaluation
* Better dashboards for inference logs and monitoring
* GPU-optimized deployment with ONNX Runtime

---

## 🧰 Tech Stack

**Backend**

* FastAPI
* Python 3.10
* Uvicorn
* segmentation-models-pytorch
* OpenCV, NumPy, pydicom
* SQLite logs

**LLM**

* Hugging Face Transformers
* PEFT (LoRA)
* Flan-T5

**Infra**

* Docker
* Render (backend)
* Netlify (frontend)
