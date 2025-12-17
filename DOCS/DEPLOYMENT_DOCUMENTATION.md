# Dokumentasi Deployment Model Machine Learning

## 📋 Daftar Isi
1. [Overview](#overview)
2. [Arsitektur Deployment Model](#arsitektur-deployment-model)
3. [Model Files](#model-files)
4. [Deployment Model ke Flask API](#deployment-model-ke-flask-api)
5. [Deployment Model ke Streamlit UI](#deployment-model-ke-streamlit-ui)
6. [Proses Prediksi](#proses-prediksi)
7. [API Request & Response](#api-request--response)

---

## Overview

Dokumentasi ini menjelaskan bagaimana **ML Model (Random Forest Regressor)** di-deploy ke:
1. **Flask REST API** - sebagai backend service
2. **Streamlit UI** - sebagai frontend dengan fallback local prediction

Model yang digunakan:
- **Algorithm**: Random Forest Regressor
- **Target**: Log-transformed price (untuk menangani skewness)
- **Features**: carat, cut, color, clarity, table

---

## Arsitektur Deployment Model

```
┌─────────────────────────────────────────────────────────────────┐
│                        ML MODEL FILES                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  model.pkl   │  │ encoder.pkl  │  │ features.pkl │          │
│  │  (64 MB)     │  │ (Label Enc)  │  │ (Col Names)  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└──────────────────────────────────────┬──────────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
                    ▼                                     ▼
        ┌─────────────────────┐              ┌─────────────────────┐
        │    FLASK API        │              │    STREAMLIT UI     │
        │    (api.py)         │              │    (app.py)         │
        │                     │              │                     │
        │  - Load model.pkl   │              │  - Call API first   │
        │  - Load encoder.pkl │              │  - Fallback: load   │
        │  - Predict via API  │              │    model locally    │
        └─────────────────────┘              └─────────────────────┘
```

---

## Model Files

### 1. model.pkl (64 MB)
```python
# Training code (simplified)
from sklearn.ensemble import RandomForestRegressor
import joblib

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)  # y = log(price)

# Save model
joblib.dump(model, 'model.pkl')
```

### 2. encoder.pkl
```python
# Encode categorical features
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()
encoder.fit(df[['cut', 'color', 'clarity']])

# Save encoder
joblib.dump(encoder, 'encoder.pkl')
```

### 3. features.pkl
```python
# Save feature column names for consistent ordering
features = ['carat', 'cut', 'color', 'clarity', 'table']
joblib.dump(features, 'features.pkl')
```

---

## Deployment Model ke Flask API

### Langkah 1: Load Model saat Startup

```python
# api.py
import joblib
import numpy as np
import pandas as pd

# Global variables untuk model
model = None
encoder = None
features = None

def load_model():
    """Load ML model, encoder, dan features"""
    global model, encoder, features
    try:
        model = joblib.load('model.pkl')
        encoder = joblib.load('encoder.pkl')
        features = joblib.load('features.pkl')
        print("✅ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

# Load model at module level (penting untuk gunicorn!)
print("🔄 Loading model at startup...")
load_model()
```

### Langkah 2: Buat Endpoint Prediksi

```python
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS untuk akses dari Streamlit

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Ambil data dari request
    data = request.get_json()
    
    carat = float(data['carat'])
    cut = data['cut']
    color = data['color']
    clarity = data['clarity']
    table = float(data['table'])
    
    # 2. Encode fitur kategorikal
    categorical_data = [[cut, color, clarity]]
    encoded = encoder.transform(categorical_data)
    
    # 3. Buat input DataFrame
    input_data = pd.DataFrame({
        'carat': [carat],
        'cut': [encoded[0][0]],
        'color': [encoded[0][1]],
        'clarity': [encoded[0][2]],
        'table': [table]
    })
    
    # 4. Reorder columns sesuai training
    input_data = input_data[features]
    
    # 5. Prediksi (model memprediksi log price)
    log_price = model.predict(input_data)[0]
    
    # 6. Transform balik ke harga asli
    price_usd = float(np.exp(log_price))
    price_idr = price_usd * 15500  # Kurs USD ke IDR
    
    # 7. Return response
    return jsonify({
        "success": True,
        "prediction": {
            "price_usd": round(price_usd, 2),
            "price_idr": round(price_idr, 0)
        }
    })
```

### Langkah 3: Jalankan API

```bash
# Development
python api.py

# Production (dengan gunicorn)
gunicorn api:app --bind 0.0.0.0:5000
```

---

## Deployment Model ke Streamlit UI

### Langkah 1: Load Model (Fallback)

```python
# app.py
import streamlit as st
import joblib
import requests

# API URL (bisa dari environment variable)
API_URL = os.environ.get('API_URL', 'https://api.example.com')

# Load model lokal sebagai fallback
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.pkl')
        encoder = joblib.load('encoder.pkl')
        features = joblib.load('features.pkl')
        return model, encoder, features
    except:
        return None, None, None
```

### Langkah 2: Buat Fungsi Prediksi via API

```python
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

### Langkah 3: Fallback ke Prediksi Lokal

```python
def predict_price_local(model, encoder, features, carat, cut, color, clarity, table):
    """Prediksi harga menggunakan model lokal (fallback)"""
    categorical_data = [[cut, color, clarity]]
    encoded = encoder.transform(categorical_data)
    
    input_data = pd.DataFrame({
        'carat': [carat],
        'cut': [encoded[0][0]],
        'color': [encoded[0][1]],
        'clarity': [encoded[0][2]],
        'table': [table]
    })
    
    input_data = input_data[features]
    log_price = model.predict(input_data)[0]
    price = np.exp(log_price)
    return price

def predict_price(model, encoder, features, carat, cut, color, clarity, table):
    """Prediksi harga - coba API dulu, fallback ke lokal"""
    # Coba prediksi via API
    price, success = predict_price_api(carat, cut, color, clarity, table)
    if success:
        return price
    
    # Fallback ke prediksi lokal
    if model is not None:
        return predict_price_local(model, encoder, features, carat, cut, color, clarity, table)
    
    raise Exception("Tidak bisa melakukan prediksi")
```

### Langkah 4: Jalankan Streamlit

```bash
streamlit run app.py
```

---

## Proses Prediksi

### Flow Prediksi (Step by Step)

```
INPUT USER                 PREPROCESSING              MODEL PREDICTION
┌──────────────┐          ┌──────────────┐           ┌──────────────┐
│ carat: 0.5   │          │ Encode cut   │           │ Random Forest│
│ cut: Ideal   │  ──────▶ │ Encode color │  ──────▶  │ Predict      │
│ color: F     │          │ Encode clrty │           │ log(price)   │
│ clarity: VS1 │          │ Create DF    │           │              │
│ table: 57    │          └──────────────┘           └──────────────┘
└──────────────┘                                            │
                                                            ▼
                                                   ┌──────────────┐
OUTPUT                     POSTPROCESSING          │ log_price    │
┌──────────────┐          ┌──────────────┐         │ = 7.44       │
│ $1,714.64    │  ◀────── │ exp(log_p)   │ ◀────── └──────────────┘
│ Rp 26.5 juta │          │ * kurs IDR   │
└──────────────┘          └──────────────┘
```

### Encoding Categorical Features

| Feature | Values | Encoded |
|---------|--------|---------|
| cut | Fair, Good, Very Good, Premium, Ideal | 0, 1, 2, 3, 4 |
| color | J, I, H, G, F, E, D | 0, 1, 2, 3, 4, 5, 6 |
| clarity | I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF | 0, 1, 2, 3, 4, 5, 6, 7 |

---

## API Request & Response

### Request

```http
POST /predict HTTP/1.1
Host: localhost:5000
Content-Type: application/json

{
    "carat": 0.5,
    "cut": "Ideal",
    "color": "F",
    "clarity": "VS1",
    "table": 57.0
}
```

### Response (Success)

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

### Response (Error)

```json
{
    "success": false,
    "error": "Missing required fields: carat, cut"
}
```

---

## Summary

| Komponen | Teknologi | Fungsi |
|----------|-----------|--------|
| **Model** | Random Forest Regressor | Prediksi harga diamond |
| **Encoder** | OrdinalEncoder | Encode fitur kategorikal |
| **API** | Flask + gunicorn | Serve model sebagai REST API |
| **UI** | Streamlit | Frontend interaktif |
| **Serialization** | joblib | Save/load model |

---

*Dokumentasi Model Deployment untuk UAS Machine Learning*
