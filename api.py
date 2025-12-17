"""
Diamond Price Prediction - Flask REST API
Backend API untuk prediksi harga diamond menggunakan ML model.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)  # Enable CORS untuk Streamlit

# Load model dan encoder
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

# Opsi valid untuk fitur kategorikal
VALID_CUTS = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
VALID_COLORS = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
VALID_CLARITIES = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

# Exchange rate
USD_TO_IDR = 15500


@app.route('/')
def home():
    """Welcome endpoint"""
    return jsonify({
        "message": "Diamond Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "This welcome message",
            "GET /health": "Health check",
            "POST /predict": "Predict diamond price"
        }
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "encoder_loaded": encoder is not None,
        "features_loaded": features is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict diamond price
    
    Request Body:
    {
        "carat": float (0.2-5.0),
        "cut": string (Fair|Good|Very Good|Premium|Ideal),
        "color": string (J|I|H|G|F|E|D),
        "clarity": string (I1|SI2|SI1|VS2|VS1|VVS2|VVS1|IF),
        "table": float (43-95)
    }
    """
    try:
        # Check if model is loaded
        if model is None or encoder is None or features is None:
            return jsonify({
                "success": False,
                "error": "Model not loaded. Please check server logs."
            }), 500
        
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided"
            }), 400
        
        # Validate required fields
        required_fields = ['carat', 'cut', 'color', 'clarity', 'table']
        missing_fields = [f for f in required_fields if f not in data]
        
        if missing_fields:
            return jsonify({
                "success": False,
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        # Extract and validate values
        carat = float(data['carat'])
        cut = data['cut']
        color = data['color']
        clarity = data['clarity']
        table = float(data['table'])
        
        # Validate ranges
        if not (0.2 <= carat <= 5.0):
            return jsonify({
                "success": False,
                "error": "Carat must be between 0.2 and 5.0"
            }), 400
        
        if cut not in VALID_CUTS:
            return jsonify({
                "success": False,
                "error": f"Invalid cut. Must be one of: {', '.join(VALID_CUTS)}"
            }), 400
        
        if color not in VALID_COLORS:
            return jsonify({
                "success": False,
                "error": f"Invalid color. Must be one of: {', '.join(VALID_COLORS)}"
            }), 400
        
        if clarity not in VALID_CLARITIES:
            return jsonify({
                "success": False,
                "error": f"Invalid clarity. Must be one of: {', '.join(VALID_CLARITIES)}"
            }), 400
        
        if not (43.0 <= table <= 95.0):
            return jsonify({
                "success": False,
                "error": "Table must be between 43 and 95"
            }), 400
        
        # Encode categorical features
        categorical_data = [[cut, color, clarity]]
        encoded = encoder.transform(categorical_data)
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'carat': [carat],
            'cut': [encoded[0][0]],
            'color': [encoded[0][1]],
            'clarity': [encoded[0][2]],
            'table': [table]
        })
        
        # Reorder columns to match training features
        input_data = input_data[features]
        
        # Predict (model predicts log price)
        log_price = model.predict(input_data)[0]
        price_usd = float(np.exp(log_price))
        price_idr = price_usd * USD_TO_IDR
        
        return jsonify({
            "success": True,
            "prediction": {
                "price_usd": round(price_usd, 2),
                "price_idr": round(price_idr, 0)
            },
            "input": {
                "carat": carat,
                "cut": cut,
                "color": color,
                "clarity": clarity,
                "table": table
            }
        })
        
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": f"Invalid value: {str(e)}"
        }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Prediction error: {str(e)}"
        }), 500


if __name__ == '__main__':
    # Load model saat startup
    if load_model():
        print("🚀 Starting Diamond Price Prediction API...")
        print("📍 Endpoints:")
        print("   GET  /        - Welcome")
        print("   GET  /health  - Health check")
        print("   POST /predict - Predict diamond price")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ Failed to load model. Exiting.")
