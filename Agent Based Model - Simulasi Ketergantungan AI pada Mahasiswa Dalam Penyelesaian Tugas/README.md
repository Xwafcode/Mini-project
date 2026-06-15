# Agent Based Model - Simulasi Ketergantungan AI pada Mahasiswa Dalam Penyelesaian Tugas

Simulasi Agent-Based Model (ABM) menggunakan framework Mesa yang memodelkan dinamika ketergantungan mahasiswa informatika terhadap Generative AI dalam penyelesaian tugas selama 16 minggu perkuliahan.

## Fitur

- **Simulasi Utama** — Visualisasi rata-rata ketergantungan AI, distribusi status mahasiswa, rentang min-maks, dan histogram.
- **Komparasi 4 Skenario** — Tanpa Intervensi, Reaktif, Preventif, dan FOMO Tinggi.
- **Monte Carlo** — Analisis statistik dengan multiple iterasi simulasi.
- **Eksplorasi Agen** — Trajektori individual mahasiswa dan scatter plot R vs K.

## Model

Setiap agen (mahasiswa) memiliki parameter:
- **K** — Level Ketergantungan AI (0–1)
- **T** — Kesulitan Tugas & Deadline
- **F** — FOMO / Pengaruh Teman
- **R** — Pemahaman Fundamental
- **D** — Deteksi Dosen

Formula update:
```
K(t+1) = K(t) + (T × F) - (R × D × K)
```

## Cara Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

- Python
- Streamlit
- Mesa (Agent-Based Modeling)
- Pandas, NumPy, Matplotlib
