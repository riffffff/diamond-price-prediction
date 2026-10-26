# Diamond Price Prediction - Aplikasi Prediksi Harga Berlian Berbasis Machine Learning dan Web

Diamond Price Prediction adalah aplikasi prediksi harga berlian berbasis web dan REST API yang dibangun menggunakan bahasa pemrograman Python. Aplikasi ini mengimplementasikan algoritma Machine Learning (Random Forest Regressor) serta pola arsitektur Flask REST API dan Streamlit Frontend untuk mengelola prediksi data tunggal maupun perbandingan harga berlian.

## Informasi Akademik / Tugas Kuliah

* **Mata Kuliah**: [Nama Mata Kuliah, contoh: Machine Learning / Pemrograman Web Terintegrasi]
* **Dosen Pengampu**: [Nama Dosen Pengampu]
* **Program Studi / Jurusan**: [Nama Program Studi]
* **Instansi / Universitas**: [Nama Perguruan Tinggi]

### Anggota Kelompok / Pengembang

| Nama Mahasiswa | NIM | Peran |
| --- | --- | --- |
| [Nama Anda / Anggota 1] | [NIM 1] | Lead Developer / ML Engineer |
| [Nama Anggota 2] | [NIM 2] | Frontend / Backend Developer |

---

## Daftar Isi

* [Fitur Utama](#fitur-utama)
* [Arsitektur dan Teknologi](#arsitektur-dan-teknologi)
* [Struktur Proyek](#struktur-proyek)
* [Persyaratan Sistem](#persyaratan-sistem)
* [Instalasi dan Penggunaan](#instalasi-dan-penggunaan)
* [Integrasi REST API](#integrasi-rest-api)

## Fitur Utama

* **Prediksi Harga Tunggal**: Memprediksi harga estimasi berlian dalam mata uang USD dan IDR berdasarkan kriteria fisik (carat, cut, color, clarity, table).
* **Perbandingan Harga Berlian**: Membandingkan dua opsi berlian secara berdampingan untuk menghitung selisih nominal dan persentase harga.
* **Backend REST API**: Menyediakan endpoint API berbasis Flask untuk integrasi prediksi harga dengan aplikasi eksternal.
* **Sistem Fallback Otomatis**: Aplikasi Streamlit dapat beralih menggunakan model lokal apabila layanan REST API remote tidak terjangkau.
* **Validasi Input & Enkodasi**: Pengolahan data otomatis untuk validasi batas rentang angka dan transformator data kategorikal.

## Arsitektur dan Teknologi

Aplikasi ini menggunakan teknologi dan pustaka standar dalam pengembangan Machine Learning dan Web Services:

* **Bahasa Pemrograman**: Python 3.10
* **Machine Learning & Data Processing**: Scikit-Learn, Pandas, NumPy, Joblib
* **Backend REST API**: Flask, Flask-CORS, Gunicorn
* **Antarmuka Web (Frontend)**: Streamlit, Requests
* **Deployment & Containerization**: Docker, Git LFS, Hugging Face Hub

## Struktur Proyek

```
riffffff-diamond-price-prediction/
├── api.py                    # Server REST API Flask dan logika inference
├── API_README.md             # Panduan ringkas penggunaan REST API
├── app.py                    # Antarmuka web interaktif berbasis Streamlit
├── Dockerfile                 # Konfigurasi pembungkusan container Docker
├── encoder.pkl               # File pkl transformer fitur kategorikal
├── features.pkl              # File pkl daftar nama fitur model ML
├── model.pkl                 # File pkl model Random Forest Regressor
├── requirements.txt          # Daftar dependensi pustaka Python
└── DOCS/
    └── DEPLOYMENT_DOCUMENTATION.md  # Dokumentasi rinci prosedur deployment
```

## Persyaratan Sistem

* **Python**: Versi 3.10 atau lebih baru
* **Package Manager**: `pip` versi terbaru
* **Container Runtime (Opsional)**: Docker Engine 20.10+
* **Koneksi Internet**: Diperlukan saat startup awal jika file model diunduh otomatis dari Hugging Face Hub

## Instalasi dan Penggunaan

1. Clone repositori ini ke komputer lokal Anda:
   ```bash
   git clone <URL_REPOSITORY>
   cd riffffff-diamond-price-prediction
   ```
2. Buat dan aktifkan lingkungan virtual (Virtual Environment):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Untuk Linux/macOS
   # venv\Scripts\activate   # Untuk Windows
   ```
3. Instal seluruh dependensi yang dibutuhkan:
   ```bash
   pip install -r requirements.txt
   ```
4. Jalankan server REST API:
   ```bash
   python api.py
   ```
   API akan berjalan secara lokal di `http://localhost:5000`.
5. Buka terminal baru dan jalankan aplikasi web Streamlit:
   ```bash
   streamlit run app.py
   ```
   Tampilan web akan otomatis terbuka di browser pada `http://localhost:8501`.

## Integrasi REST API

Layanan REST API dapat diakses secara lokal maupun melalui endpoint yang terdeploy di Hugging Face Spaces (`https://rifaifirdaus-diamond-prediction-api.hf.space`).

| Method | Endpoint | Deskripsi |
| --- | --- | --- |
| `GET` | `/` | Menampilkan pesan selamat datang dan daftar endpoint aktif |
| `GET` | `/health` | Memeriksa status kesehatan server dan muatan model ML |
| `POST` | `/predict` | Memprediksi harga berlian berdasarkan parameter JSON |

Contoh format request JSON untuk endpoint `/predict`:

```json
{
  "carat": 0.5,
  "cut": "Ideal",
  "color": "F",
  "clarity": "VS1",
  "table": 57.0
}
```