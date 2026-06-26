# HealthCost AI — Prediksi Biaya Asuransi (Regression Insurance)

Proyek ini merupakan aplikasi berbasis kecerdasan buatan (AI) menggunakan model Machine Learning (Random Forest Regression) untuk memprediksi perkiraan biaya asuransi kesehatan berdasarkan berbagai faktor seperti usia, BMI, jumlah anak, dan kebiasaan merokok.

## Struktur Direktori

- `01_eda.ipynb`: Exploratory Data Analysis (EDA) untuk memahami distribusi dan korelasi antar fitur.
- `02_data_cleaning.ipynb`: Pembersihan data dan pra-pemrosesan.
- `03_model_building.ipynb`: Pelatihan dan evaluasi model regresi (Random Forest).
- `app.py`: Aplikasi antarmuka berbasis Streamlit dengan UI yang modern dan interaktif.
- `Data/`: Berisi dataset mentah dan hasil pembersihan (`insurance.csv`, `insurance_clean.csv`).
- `model/`: Artefak model Machine Learning yang telah dilatih (`rf_model.pkl`, `columns.pkl`).

## Cara Menjalankan Aplikasi

1. Pastikan library yang dibutuhkan telah terinstall (Streamlit, Pandas, NumPy, Scikit-Learn).
2. Jalankan perintah berikut di dalam direktori proyek ini:

```bash
streamlit run app.py
```
3. Akses antarmuka aplikasi melalui tautan lokal yang muncul di terminal (default: `http://localhost:8501`).
