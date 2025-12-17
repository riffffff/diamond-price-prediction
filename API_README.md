# Diamond Price Prediction API

REST API untuk prediksi harga diamond menggunakan Machine Learning.

## 🚀 Endpoints

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| GET | `/` | Welcome message |
| GET | `/health` | Health check |
| POST | `/predict` | Prediksi harga diamond |

## 📝 Request & Response

### POST /predict

**Request:**
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
  "input": { ... }
}
```

## 🔧 Parameter Validation

| Parameter | Type | Range |
|-----------|------|-------|
| carat | float | 0.2 - 5.0 |
| cut | string | Fair, Good, Very Good, Premium, Ideal |
| color | string | J, I, H, G, F, E, D |
| clarity | string | I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF |
| table | float | 43.0 - 95.0 |

## 🏃 Menjalankan Lokal

```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan API
python3 api.py

# API berjalan di http://localhost:5000
```

## 🌐 Deploy ke Render

1. Push ke GitHub
2. Buat Web Service di [render.com](https://render.com)
3. Connect ke repo GitHub
4. Settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn api:app --bind 0.0.0.0:$PORT`

## 🔗 Integrasi dengan Streamlit

Set environment variable `API_URL` di Streamlit Cloud:
```
API_URL=https://your-api.onrender.com
```
