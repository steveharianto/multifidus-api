# Multifidus MRI Segmentation Demo

End-to-end AI service that segments the lumbar multifidus muscle on axial T2 MRI slices and exposes the model through a FastAPI REST API.  
Originally this project also included a LoRA-fine-tuned language model; however, due to memory limits on the Render free tier, the **deployed backend contains only the segmentation service**.  
The LoRA model and training code remain in the repository as a demonstration of LLM fine-tuning capability.

This project demonstrates **full-stack AI engineering capability**:
- CV model training + preprocessing  
- API deployment (Docker + FastAPI)  
- Frontend web demo (TailwindCSS + static hosting)  
- Production-style logging, database, and containerized architecture  
- LLM fine-tuning (LoRA) included in repo (not deployed to cloud)

---

## 🚀 Live Demo

- **Frontend (Netlify):** https://multifidus-demo.netlify.app  
- **Backend Segmentation API (Render):** https://multifidus-api.onrender.com/docs  

---

## 📌 What This Project Does

### Deployed Demo
- Accepts a single axial T2-weighted lumbar MRI slice (`.dcm` or `.png/.jpg`).
- Runs a U-Net (ResNet-34 encoder) segmentation model to highlight the multifidus muscle.
- Returns a color overlay as a base64 PNG.
- Logs inference latency into SQLite inside the container.
- Hosted as a Dockerized FastAPI service on Render.

### Included in Repository (not deployed)
- LoRA-fine-tuned Flan-T5 model that generates clinical-style summaries from numerical measurements.  
  This demonstrates LLM fine-tuning ability even though the actual model cannot be deployed on a 512 MB service.

---

## 🏗️ Components

* **Frontend:** Static TailwindCSS single-page web app (Netlify).
* **Backend:** FastAPI + Uvicorn inside a Docker container (Render).
* **Segmentation model:** U-Net with ResNet-34 encoder.
* **Logging:** SQLite database for inference logs.
* **LLM module (local):** Flan-T5 with LoRA adapters (repo only).

---

## 🔄 Pipeline Details

### 🧠 Segmentation Pipeline (Deployed)

1. **Input formats**
   - `.dcm` via `pydicom`
   - `.png` / `.jpg` grayscale via OpenCV

2. **Preprocessing**
   - Convert to float32  
   - Min–max normalize to `[0,1]`  
   - Resize to `256×256`  
   - Shape `(1,1,H,W)`  

3. **Model**
   - U-Net with ResNet-34 encoder  
   - 1 output class → sigmoid → threshold at 0.5  

4. **Postprocessing**
   - Resize mask back to original size  
   - Apply JET colormap  
   - Alpha-blend with original slice  
   - Encode as PNG → base64  
   - Log latency into SQLite  

---

### 📝 LoRA Reporting Pipeline (Not Deployed)

The repository contains a complete training and inference script for a small LoRA fine-tuned model:

1. **Input fields**
   - `level`  
   - `side`  
   - `muscle_area_mm2`  
   - `fat_infiltration_pct`  
   - `degeneration_grade`  

2. **Model**
   - `google/flan-t5-base`  
   - LoRA on attention projections (`q`, `v`)  
   - Trainable params ~0.36% of base model  

3. **Output format**
   - `Summary:`  
   - `Risk:`  
   - `Recommendation:`  

**Note:** This model runs correctly in Colab/local, but is not deployed due to Render’s RAM limitations.

---

## 🧩 Models & Data

### 🔹 Segmentation Model (U-Net ResNet-34)

- **Parameters:** ~24.8M  
- **Training data:** Axial T2 MRI slices with manual ITK-Snap segmentation  
- **Training setup:**
  - Slice-based training  
  - 256×256 resolution  
  - Augmentations: flips, light rotation  
- **Loss:** Binary cross-entropy with sigmoid  

---

### 🔹 LoRA Model (Flan-T5) — Repository Only

- **Base:** 248M parameter model  
- **Fine-tuning:** LoRA `r=8`, dropout 0.1  
- **Dataset:** ~20 handcrafted clinical-style examples  
- **Purpose:** Demonstrate LLM engineering skills  

---

## 📡 API Endpoints (Deployed)

### ➤ `GET /health`
Simple health check.

---

### ➤ `POST /segment`
Run segmentation on an uploaded MRI slice.

#### Request
```

Content-Type: multipart/form-data
file: <MRI slice>

````

#### Response
```json
{
  "filename": "slice01.dcm",
  "latency_ms": 88,
  "overlay_base64": "<base64PNG>",
  "width": 512,
  "height": 512
}
````

---

## 📝 API Endpoints (Local Only)

### ➤ `POST /report`

Only available in the development version that loads the LoRA model.
Not deployed to Render.

---

## 🎨 Frontend Demo (TailwindCSS)

Features:

* Drag-and-drop or file upload
* Clean medical-themed UI
* Real-time calls to `/segment`
* Loading and error states
* Shows inference latency
* Deployed via Netlify

URL: **[https://multifidus-demo.netlify.app](https://multifidus-demo.netlify.app)**

---

## 🛠️ Local Development

### 1) Clone

```bash
git clone https://github.com/<your-username>/multifidus-api.git
cd multifidus-api
```

### 2) Install (segmentation backend)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Run the API

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open Swagger UI: `http://localhost:8000/docs`

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

Backend now available at:
`http://localhost:8000`

---

## ☁️ Cloud Deployment

### Backend (Render)

* Uses a slim Python base image
* Installs OpenCV dependencies
* Loads only the segmentation model (fits in 512 MB RAM)
* Startup command:

```
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Frontend (Netlify)

* Static Tailwind build
* Calls the Render backend via fetch
* Publicly accessible demo page

---

## ⚠️ Limitations & Future Work

### Current Limitations

* Not a medical device
* Single-slice segmentation only
* No GPU acceleration on Render
* LoRA model not deployed due to memory limits

### Future Enhancements

* Automatic spinal level detection
* 3D segmentation
* Deployment with GPU-backed inference
* ONNX/TensorRT optimization
* Deploy LoRA reporter on a higher-tier GPU cloud

---

## 🧰 Tech Stack

### Backend

* FastAPI
* Python 3.10
* Uvicorn
* segmentation-models-pytorch
* OpenCV, NumPy, pydicom
* SQLite logging

### LLM (local)

* Hugging Face Transformers
* PEFT (LoRA)
* Flan-T5

### Infra

* Docker
* Render (backend)
* Netlify (frontend)
