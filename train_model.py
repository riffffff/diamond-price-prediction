"""
Diamond Price Prediction - Model Training Script
Kelompok 4 - UAS Praktikum Penambangan Data

Script ini melatih model Random Forest untuk memprediksi harga diamond
dan menyimpan model beserta encoder ke file pickle.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# URL dataset
DATA_URL = 'https://raw.githubusercontent.com/rifaifirdaus/UTS_Praktikum_Penambangan_Data/refs/heads/main/diamonds.csv'

def load_and_preprocess_data():
    """Load dataset dan lakukan preprocessing"""
    print("Loading dataset...")
    df = pd.read_csv(DATA_URL)
    
    # Drop kolom Unnamed: 0 (index)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # Hapus baris dengan nilai 0 pada x, y, z
    df = df[(df['x'] > 0) & (df['y'] > 0) & (df['z'] > 0)]
    
    print(f"Dataset shape: {df.shape}")
    return df

def encode_categorical_features(df):
    """Encode fitur kategorikal menggunakan OrdinalEncoder"""
    # Definisikan urutan untuk ordinal encoding
    cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    color_order = ['J', 'I', 'H', 'G', 'F', 'E', 'D']  # J worst, D best
    clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']  # I1 worst, IF best
    
    encoder = OrdinalEncoder(categories=[cut_order, color_order, clarity_order])
    
    categorical_cols = ['cut', 'color', 'clarity']
    df_encoded = df.copy()
    df_encoded[categorical_cols] = encoder.fit_transform(df[categorical_cols])
    
    return df_encoded, encoder

def train_model(df):
    """Train Random Forest model"""
    print("\nTraining model...")
    
    # Encode categorical features
    df_encoded, encoder = encode_categorical_features(df)
    
    # Log transform price (target)
    df_encoded['price'] = np.log(df_encoded['price'])
    
    # Hapus fitur yang redundan (x, y, z, depth)
    # Karena x, y, z sangat berkorelasi dengan carat
    # dan depth tidak signifikan
    cols_to_drop = ['price', 'x', 'y', 'z', 'depth']
    X = df_encoded.drop(columns=cols_to_drop)
    y = df_encoded['price']
    
    print(f"Features used: {X.columns.tolist()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest (model lebih kecil untuk deployment)
    model = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    from sklearn.metrics import mean_squared_error, r2_score
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"MSE : {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R²  : {r2:.6f}")
    
    # Feature importance
    print(f"\nFeature Importance:")
    for feat, imp in sorted(zip(X.columns, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f} ({imp*100:.1f}%)")
    
    return model, encoder, X.columns.tolist()

def save_model(model, encoder, feature_names, output_dir='.'):
    """Simpan model dan encoder ke file"""
    model_path = os.path.join(output_dir, 'model.pkl')
    encoder_path = os.path.join(output_dir, 'encoder.pkl')
    features_path = os.path.join(output_dir, 'features.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)
    joblib.dump(feature_names, features_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Encoder saved to: {encoder_path}")
    print(f"Features saved to: {features_path}")

def main():
    # Load data
    df = load_and_preprocess_data()
    
    # Train model
    model, encoder, feature_names = train_model(df)
    
    # Save model
    save_model(model, encoder, feature_names)
    
    print("\n✅ Training complete!")

if __name__ == "__main__":
    main()
