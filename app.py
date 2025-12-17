"""
Diamond Price Prediction - Streamlit App
Aplikasi web untuk memprediksi harga diamond menggunakan
model Machine Learning (Random Forest Regressor).
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import requests

# Konfigurasi halaman
st.set_page_config(
    page_title="Diamond Price Prediction",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern Dark Mode CSS tanpa gradient
st.markdown("""
<style>
    /* Force Dark Mode */
    :root {
        color-scheme: dark;
    }
    
    .stApp {
        background: #0a0a0f;
        color: #e0e0e0;
    }
    
    /* Override light mode elements */
    .stApp, .main, section[data-testid="stSidebar"], 
    .stMarkdown, .stText, p, span, label, .stRadio label,
    div[data-testid="stForm"], .stExpander {
        color: #e0e0e0 !important;
    }
    
    /* Splash screen styles */
    .splash-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 60vh;
        animation: fadeIn 0.5s ease-in;
    }
    
    .splash-diamond {
        font-size: 5rem;
        animation: float 3s ease-in-out infinite;
    }
    
    .splash-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #a855f7;
        margin: 1rem 0;
        animation: slideUp 0.8s ease-out;
    }
    
    .splash-subtitle {
        color: #6b7280;
        font-size: 1.1rem;
        animation: slideUp 1s ease-out;
    }
    
    .splash-loading {
        margin-top: 2rem;
        display: flex;
        gap: 0.5rem;
    }
    
    .splash-dot {
        width: 10px;
        height: 10px;
        background: #a855f7;
        border-radius: 50%;
        animation: bounce 1.4s ease-in-out infinite;
    }
    
    .splash-dot:nth-child(1) { animation-delay: 0s; }
    .splash-dot:nth-child(2) { animation-delay: 0.2s; }
    .splash-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes bounce {
        0%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-8px); }
    }
    
    /* Main header styles */
    .main-header {
        text-align: center;
        padding: 1rem 0 1.5rem 0;
    }
    .main-header h1 {
        font-size: 2rem;
        margin: 0;
        color: #ffffff;
        font-weight: 600;
        letter-spacing: -0.5px;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    .main-header .diamond-icon {
        font-size: 1.8rem;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    
    /* Form title */
    .form-title {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        color: #e5e7eb;
    }
    .form-title.a { color: #60a5fa; }
    .form-title.b { color: #a855f7; }
    
    /* Container styling */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
    }
    
    /* Input styling */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #e0e0e0 !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus,
    .stNumberInput > div > div > input:hover,
    .stSelectbox > div > div > div:hover {
        border-color: #a855f7 !important;
        background: rgba(168, 85, 247, 0.05) !important;
    }
    
    .stNumberInput label, .stSelectbox label {
        color: #9ca3af !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }
    
    /* Range info text - smaller font */
    .range-info {
        font-size: 0.55rem;
        color: #4b5563;
        margin-top: -0.6rem;
        margin-bottom: 0.3rem;
    }
    
    /* Result card - clean without gradient */
    .result-card {
        background: #18181b;
        border: 1px solid rgba(168, 85, 247, 0.3);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
        scroll-margin-top: 20px;
    }
    .result-card .label {
        font-size: 0.85rem;
        color: #a855f7;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    .result-card .price {
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0.5rem 0;
    }
    .result-card .price-idr {
        font-size: 1.2rem;
        color: #9ca3af;
        margin-top: 0.3rem;
    }
    
    /* Comparison result cards */
    .compare-card {
        background: #18181b;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.06);
        transition: all 0.2s ease;
    }
    .compare-card:hover {
        border-color: rgba(255, 255, 255, 0.12);
    }
    .compare-card.a { 
        border-color: rgba(96, 165, 250, 0.3);
    }
    .compare-card.b { 
        border-color: rgba(168, 85, 247, 0.3);
    }
    .compare-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    .compare-label.a { color: #60a5fa; }
    .compare-label.b { color: #a855f7; }
    .compare-price {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
    }
    .compare-price-idr {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-top: 0.2rem;
    }
    
    .diff-card {
        background: #18181b;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .diff-label {
        font-size: 0.75rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .diff-amount {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .diff-idr {
        font-size: 0.75rem;
        color: #6b7280;
    }
    .diff-text {
        font-size: 0.85rem;
        color: #9ca3af;
    }
    
    /* Comparison results wrapper */
    .compare-results {
        scroll-margin-top: 20px;
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        margin: 1.5rem 0;
    }
    
    /* Button styling - clean */
    .stButton > button {
        background: #a855f7 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: #9333ea !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* Radio button styling */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.03);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    .stRadio > div label {
        color: #9ca3af !important;
    }
    
    .stRadio > div [data-testid="stMarkdownContainer"] p {
        color: #e5e7eb !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #4b5563;
        font-size: 0.75rem;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.04);
    }
    
    /* Hide hamburger menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Detail table styling */
    .detail-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 1rem;
    }
    .detail-table th, .detail-table td {
        padding: 0.75rem 1rem;
        text-align: left;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    .detail-table th {
        color: #6b7280;
        font-weight: 500;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .detail-table td {
        color: #e5e7eb;
    }
    .detail-table tr:last-child td {
        border-bottom: none;
    }
    
    /* Smooth scroll */
    html {
        scroll-behavior: smooth;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
# Set API_URL via environment variable atau gunakan default HF Spaces
API_URL = os.environ.get('API_URL', 'https://rifaifirdaus-diamond-prediction-api.hf.space')

# Load model dan encoder (fallback jika API tidak tersedia)
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.pkl')
        encoder = joblib.load('encoder.pkl')
        features = joblib.load('features.pkl')
        return model, encoder, features
    except:
        return None, None, None

# Opsi untuk fitur kategorikal
CUT_OPTIONS = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
COLOR_OPTIONS = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
CLARITY_OPTIONS = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

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
    
    # Jika keduanya gagal
    raise Exception("Tidak bisa melakukan prediksi. API tidak tersedia dan model lokal tidak ditemukan.")

def render_compact_form(prefix=""):
    """Form input untuk karakteristik diamond"""
    
    # Row 1: Carat & Cut
    col1, col2 = st.columns(2)
    with col1:
        carat = st.number_input("Carat", min_value=0.2, max_value=5.0, value=0.5, step=0.01, key=f"{prefix}carat")
        st.markdown('<p class="range-info">0.2 - 5.0 ct</p>', unsafe_allow_html=True)
    with col2:
        cut = st.selectbox("Cut", options=CUT_OPTIONS, index=4, key=f"{prefix}cut")
        st.markdown('<p class="range-info">Fair - Ideal (terbaik)</p>', unsafe_allow_html=True)
    
    # Row 2: Color & Clarity
    col3, col4 = st.columns(2)
    with col3:
        color = st.selectbox("Color", options=COLOR_OPTIONS, index=4, key=f"{prefix}color")
        st.markdown('<p class="range-info">J (terburuk) - D (terbaik)</p>', unsafe_allow_html=True)
    with col4:
        clarity = st.selectbox("Clarity", options=CLARITY_OPTIONS, index=4, key=f"{prefix}clarity")
        st.markdown('<p class="range-info">I1 (terburuk) - IF (terbaik)</p>', unsafe_allow_html=True)
    
    # Row 3: Table only
    table = st.number_input("Table", min_value=43.0, max_value=95.0, value=57.0, step=0.1, key=f"{prefix}table")
    st.markdown('<p class="range-info">43% - 95%</p>', unsafe_allow_html=True)
    
    return carat, cut, color, clarity, table

def show_splash_screen():
    """Menampilkan splash screen animasi"""
    splash = st.empty()
    
    with splash.container():
        st.markdown("""
        <div class="splash-container">
            <div class="splash-diamond">💎</div>
            <div class="splash-title">Diamond Price Prediction</div>
            <div class="splash-subtitle">Prediksi Harga Diamond dengan Machine Learning</div>
            <div class="splash-loading">
                <div class="splash-dot"></div>
                <div class="splash-dot"></div>
                <div class="splash-dot"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    time.sleep(2)
    splash.empty()

def main():
    # Session state untuk splash screen
    if 'splash_shown' not in st.session_state:
        st.session_state.splash_shown = False
    
    # Tampilkan splash screen hanya sekali
    if not st.session_state.splash_shown:
        show_splash_screen()
        st.session_state.splash_shown = True
    
    # Header dengan diamond icon
    st.markdown("""
    <div class="main-header">
        <h1><span class="diamond-icon">💎</span> Diamond Price Prediction</h1>
        <p class="subtitle">Prediksi harga diamond menggunakan Random Forest Regressor</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if model exists
    if not os.path.exists('model.pkl'):
        st.error("Model belum di-train! Jalankan python train_model.py terlebih dahulu.")
        return
    
    # Load model
    try:
        model, encoder, features = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    # Mode selector - centered
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        mode = st.radio(
            "Pilih Mode",
            options=["Prediksi Tunggal", "Perbandingan"],
            horizontal=True,
            label_visibility="collapsed"
        )
    
    st.divider()
    
    if mode == "Prediksi Tunggal":
        # Single Prediction Mode - centered form
        col_spacer1, col_form, col_spacer2 = st.columns([1, 2, 1])
        
        with col_form:
            with st.container(border=True):
                st.markdown('<div class="form-title">Karakteristik Diamond</div>', unsafe_allow_html=True)
                carat, cut, color, clarity, table = render_compact_form()
            
            st.markdown("")
            
            if st.button("Prediksi Harga", type="primary", use_container_width=True):
                price = predict_price(model, encoder, features, carat, cut, color, clarity, table)
                
                # Result card with ID for scrolling
                price_idr = price * 15500  # Kurs USD ke IDR
                st.markdown(f"""
                <div class="result-card" id="estimasi-harga">
                    <div class="label">Estimasi Harga</div>
                    <div class="price">${price:,.2f}</div>
                    <div class="price-idr">Rp {price_idr:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Detail table
                st.markdown(f"""
                <table class="detail-table">
                    <tr><th>Karakteristik</th><th>Nilai</th></tr>
                    <tr><td>Carat</td><td>{carat:.2f} ct</td></tr>
                    <tr><td>Cut</td><td>{cut}</td></tr>
                    <tr><td>Color</td><td>{color}</td></tr>
                    <tr><td>Clarity</td><td>{clarity}</td></tr>
                    <tr><td>Table</td><td>{table:.1f}%</td></tr>
                </table>
                """, unsafe_allow_html=True)
                
                # Smooth scroll to estimasi harga
                st.components.v1.html("""
                <script>
                    setTimeout(function() {
                        var parentDoc = window.parent.document;
                        var el = parentDoc.getElementById('estimasi-harga');
                        if (el) {
                            el.scrollIntoView({ behavior: 'smooth', block: 'start' });
                        }
                    }, 300);
                </script>
                """, height=0)
    
    else:
        # Comparison Mode - side by side forms with containers
        col_a, col_spacer, col_b = st.columns([1, 0.05, 1])
        
        with col_a:
            with st.container(border=True):
                st.markdown('<div class="form-title a">Diamond A</div>', unsafe_allow_html=True)
                carat_a, cut_a, color_a, clarity_a, table_a = render_compact_form("a_")
        
        with col_b:
            with st.container(border=True):
                st.markdown('<div class="form-title b">Diamond B</div>', unsafe_allow_html=True)
                carat_b, cut_b, color_b, clarity_b, table_b = render_compact_form("b_")
        
        st.markdown("")
        
        # Centered button
        col_btn_left, col_btn, col_btn_right = st.columns([1, 2, 1])
        with col_btn:
            compare_clicked = st.button("Bandingkan Harga", type="primary", use_container_width=True)
        
        if compare_clicked:
            price_a = predict_price(model, encoder, features, carat_a, cut_a, color_a, clarity_a, table_a)
            price_b = predict_price(model, encoder, features, carat_b, cut_b, color_b, clarity_b, table_b)
            
            diff = price_b - price_a
            diff_percent = ((price_b - price_a) / price_a) * 100
            price_a_idr = price_a * 15500  # Kurs USD ke IDR
            price_b_idr = price_b * 15500
            diff_idr = abs(diff) * 15500
            
            st.divider()
            
            # Results wrapper with ID for scrolling
            st.markdown('<div id="hasil-perbandingan"></div>', unsafe_allow_html=True)
            
            # Results - 3 columns
            res_col1, res_col2, res_col3 = st.columns([1, 0.8, 1])
            
            with res_col1:
                st.markdown(f"""
                <div class="compare-card a">
                    <div class="compare-label a">Diamond A</div>
                    <div class="compare-price">${price_a:,.2f}</div>
                    <div class="compare-price-idr">Rp {price_a_idr:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with res_col2:
                if diff > 0:
                    diff_text = "B lebih mahal"
                    diff_color = "#ef4444"
                elif diff < 0:
                    diff_text = "A lebih mahal"
                    diff_color = "#22c55e"
                else:
                    diff_text = "Harga sama"
                    diff_color = "#9ca3af"
                
                st.markdown(f"""
                <div class="diff-card">
                    <div class="diff-label">Selisih</div>
                    <div class="diff-amount" style="color: {diff_color};">${abs(diff):,.0f}</div>
                    <div class="diff-idr">Rp {diff_idr:,.0f}</div>
                    <div class="diff-text">{diff_text} ({diff_percent:+.1f}%)</div>
                </div>
                """, unsafe_allow_html=True)
            
            with res_col3:
                st.markdown(f"""
                <div class="compare-card b">
                    <div class="compare-label b">Diamond B</div>
                    <div class="compare-price">${price_b:,.2f}</div>
                    <div class="compare-price-idr">Rp {price_b_idr:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Comparison table
            st.markdown("")
            st.markdown(f"""
            <table class="detail-table">
                <tr><th>Karakteristik</th><th>Diamond A</th><th>Diamond B</th></tr>
                <tr><td>Carat</td><td>{carat_a:.2f} ct</td><td>{carat_b:.2f} ct</td></tr>
                <tr><td>Cut</td><td>{cut_a}</td><td>{cut_b}</td></tr>
                <tr><td>Color</td><td>{color_a}</td><td>{color_b}</td></tr>
                <tr><td>Clarity</td><td>{clarity_a}</td><td>{clarity_b}</td></tr>
                <tr><td>Table</td><td>{table_a:.1f}%</td><td>{table_b:.1f}%</td></tr>
            </table>
            """, unsafe_allow_html=True)
            
            # Smooth scroll to hasil perbandingan
            st.components.v1.html("""
            <script>
                setTimeout(function() {
                    var parentDoc = window.parent.document;
                    var el = parentDoc.getElementById('hasil-perbandingan');
                    if (el) {
                        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }
                }, 300);
            </script>
            """, height=0)
    
    # Footer
    st.markdown("""
    <div class="footer">
        Diamond Price Prediction 2025
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
