# Dokumentasi Deployment - Diamond Price Prediction API

## 📋 Daftar Isi
1. [Overview](#overview)
2. [Arsitektur Sistem](#arsitektur-sistem)
3. [File Struktur](#file-struktur)
4. [Deployment Flask API ke Hugging Face Spaces](#deployment-flask-api-ke-hugging-face-spaces)
5. [Deployment Streamlit UI](#deployment-streamlit-ui)
6. [API Endpoints](#api-endpoints)
7. [Integrasi Frontend & Backend](#integrasi-frontend--backend)

---

## Overview

Proyek ini mengimplementasikan **Machine Learning Deployment** untuk prediksi harga diamond dengan arsitektur:
- **Frontend**: Streamlit UI (hosted di Streamlit Cloud)
- **Backend**: Flask REST API (hosted di Hugging Face Spaces)
- **Model**: Random Forest Regressor

---

## Arsitektur Sistem

```
┌─────────────────────────┐     HTTPS Request     ┌─────────────────────────┐
│      STREAMLIT UI       │ ───────────────────▶  │      FLASK API          │
│   (Streamlit Cloud)     │                       │   (HF Spaces)           │
│                         │ ◀───────────────────  │                         │
│   - Form Input Diamond  │     JSON Response     │   - /predict endpoint   │
│   - Display Hasil       │                       │   - /health endpoint    │
└─────────────────────────┘                       └───────────┬─────────────┘
                                                              │
                                                              ▼
                                                  ┌─────────────────────────┐
                                                  │      ML MODEL           │
                                                  │   (Random Forest)       │
                                                  │   - model.pkl (64MB)    │
                                                  │   - encoder.pkl         │
                                                  │   - features.pkl        │
                                                  └─────────────────────────┘
```

---

## File Struktur

```
diamond-price-prediction/
├── api.py              # Flask REST API backend
├── app.py              # Streamlit UI frontend
├── Dockerfile          # Docker config untuk HF Spaces
├── Procfile            # Process config untuk deployment
├── requirements.txt    # Python dependencies
├── model.pkl           # Trained ML model (64MB)
├── encoder.pkl         # Label encoder
├── features.pkl        # Feature names
└── API_README.md       # API documentation
```

---

## Deployment Flask API ke Hugging Face Spaces

### Step 1: Buat Space di Hugging Face
1. Buka https://huggingface.co/spaces
2. Klik "Create new Space"
3. Pilih:
   - **Space name**: `diamond-prediction-api`
   - **SDK**: Docker
   - **Visibility**: Public

### Step 2: Siapkan Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install git-lfs for large model files
RUN apt-get update && apt-get install -y git-lfs && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY api.py .
COPY model.pkl .
COPY encoder.pkl .
COPY features.pkl .

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Run the Flask API
CMD ["gunicorn", "api:app", "--bind", "0.0.0.0:7860"]
```

### Step 3: Push ke HF Spaces
```bash
# Clone HF Space
git clone https://huggingface.co/spaces/USERNAME/diamond-prediction-api

# Copy files
cp api.py Dockerfile requirements.txt model.pkl encoder.pkl features.pkl ./

# Setup Git LFS untuk file besar (model.pkl > 10MB)
git lfs install
git lfs track "*.pkl"

# Push
git add .
git commit -m "Deploy Flask API"
git push origin main
```

### Step 4: Verifikasi Deployment
Setelah build selesai, API akan live di:
- **URL**: `https://USERNAME-diamond-prediction-api.hf.space`

Test dengan curl:
```bash
# Health check
curl https://rifaifirdaus-diamond-prediction-api.hf.space/health

# Prediction
curl -X POST https://rifaifirdaus-diamond-prediction-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"carat": 0.5, "cut": "Ideal", "color": "F", "clarity": "VS1", "table": 57}'
```

### Screenshot: API Health Check
![API Health Check](api_health_endpoint_1765940405484.png)

---

## Deployment Streamlit UI

### Step 1: Push ke GitHub
```bash
git add app.py requirements.txt
git commit -m "Add Streamlit app"
git push origin main
```

### Step 2: Deploy di Streamlit Cloud
1. Buka https://share.streamlit.io
2. Klik "New app"
3. Connect GitHub repo: `riffffff/diamond-price-prediction`
4. Main file: `app.py`
5. Deploy!

### Step 3: Set Environment Variable (Opsional)
Di Streamlit Cloud Settings, tambahkan:
```
API_URL=https://rifaifirdaus-diamond-prediction-api.hf.space
```

---

## API Endpoints

### GET /
Welcome message dan daftar endpoints.

**Response:**
```json
{
    "message": "Diamond Price Prediction API",
    "version": "1.0.0",
    "endpoints": {
        "GET /": "This welcome message",
        "GET /health": "Health check",
        "POST /predict": "Predict diamond price"
    }
}
```

### GET /health
Health check untuk monitoring.

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "encoder_loaded": true,
    "features_loaded": true
}
```

### POST /predict
Prediksi harga diamond.

**Request Body:**
```json
{
    "carat": 0.5,
    "cut": "Ideal",
    "color": "F",
    "clarity": "VS1",
    "table": 57.0
}
```

**Response:**
```json
{
    "success": true,
    "prediction": {
        "price_usd": 1714.64,
        "price_idr": 26576908.0
    },
    "input": {
        "carat": 0.5,
        "cut": "Ideal",
        "color": "F",
        "clarity": "VS1",
        "table": 57.0
    }
}
```

---

## Integrasi Frontend & Backend

### Kode Integrasi di Streamlit (app.py)

```python
import requests

# API Configuration
API_URL = 'https://rifaifirdaus-diamond-prediction-api.hf.space'

def predict_price_api(carat, cut, color, clarity, table):
    """Prediksi harga via Flask API"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={
                "carat": carat,
                "cut": cut,
                "color": color,
                "clarity": clarity,
                "table": table
            },
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                return data['prediction']['price_usd'], True
        return None, False
    except:
        return None, False
```

### Flow Integrasi
1. User input data diamond di Streamlit UI
2. Streamlit mengirim HTTP POST request ke Flask API
3. Flask API memproses prediksi dengan ML model
4. Response JSON dikembalikan ke Streamlit
5. Streamlit menampilkan hasil ke user

---

## 🔗 Live URLs

| Service | URL |
|---------|-----|
| **Flask API** | https://rifaifirdaus-diamond-prediction-api.hf.space |
| **GitHub Repo** | https://github.com/riffffff/diamond-price-prediction |

---

*Dokumentasi ini dibuat untuk keperluan UAS Deployment Model Machine Learning*
