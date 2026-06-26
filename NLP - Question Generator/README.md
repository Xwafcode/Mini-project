# NLP - Question Generator

Sistem **Question Generator** berbasis Natural Language Processing (NLP) menggunakan model **T5 (Text-to-Text Transfer Transformer)**. Sistem ini dirancang untuk menghasilkan pertanyaan secara otomatis berdasarkan konteks teks yang diberikan. Dilengkapi dengan antarmuka web interaktif berbasis **Streamlit** dengan desain UI *Liquid Glass* yang elegan.

## Struktur Proyek

```
├── data/
│   ├── qg_train.csv            # Dataset pelatihan (Question Generation)
│   └── qg_valid.csv            # Dataset validasi
├── QG_Model/
│   └── best_model/             # Model T5 hasil fine-tuning
├── QG_T5_Colab.ipynb           # Notebook pelatihan model (Google Colab)
├── app.py                      # Antarmuka web interaktif Streamlit
├── requirements.txt            # Daftar dependensi
└── README.md
```

## Dataset

Dataset yang digunakan untuk melatih model terdiri dari pasangan teks konteks dan pertanyaan yang relevan:
- **`qg_train.csv`**: Dataset pelatihan utama untuk melatih kemampuan pemahaman teks dan generasi pertanyaan model T5.
- **`qg_valid.csv`**: Dataset validasi untuk mengukur metrik performa dan mencegah *overfitting* selama proses *fine-tuning*.

*(Catatan: File dataset `.csv` berukuran besar dikelola menggunakan Git LFS).*

## Instalasi

Pastikan Anda telah menginstal **Python 3.8+**.

```bash
# Clone repositori
git clone https://github.com/Xwafcode/Mini-project.git
cd "Mini-project/NLP - Question Generator"

# Install semua dependensi
pip install -r requirements.txt
```

### Dependensi Utama

- `torch` & `transformers`: Untuk memuat dan menjalankan model T5 (`T5ForConditionalGeneration`, `T5Tokenizer`).
- `streamlit`: Untuk menjalankan antarmuka web interaktif.

## Cara Menjalankan

### Menjalankan Antarmuka Web (Streamlit)

```bash
streamlit run app.py
```

Aplikasi akan terbuka secara otomatis di browser pada alamat `http://localhost:8501`.

### Pelatihan Model Ulang (Opsional)

Jika Anda ingin melakukan eksperimen atau melatih ulang model dengan dataset yang ada, Anda dapat membuka dan menjalankan file notebook **`QG_T5_Colab.ipynb`** menggunakan Google Colab atau Jupyter Notebook lokal dengan dukungan GPU.
