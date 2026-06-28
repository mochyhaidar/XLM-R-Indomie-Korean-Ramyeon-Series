# Aspect-Based Sentiment Analysis (ABSA) Indomie Korean Ramyeon Series

Repositori ini berisi implementasi sistem **Deployment** untuk menguji model **XLM-RoBERTa** yang telah dilatih khusus menggunakan metode CRISP-DM untuk melakukan analisis sentimen berbasis aspek pada produk Indomie Korean Ramyeon Series (Rasa, Promosi, dan Kolaborasi).

Sistem ini dilengkapi dengan modul **Deteksi Aspek Otomatis (Lexicon-Based)** sebelum mengeksekusi prediksi sentimen, sehingga sistem dapat mengklasifikasikan data mentah secara instan.

---

## 🚀 Petunjuk Penggunaan Sistem Secara Lokal

### 1. Unduh Berkas Model (Trained Weights)
Karena ukuran berkas model yang besar (sekitar 1.1 GB) melampaui batas unggah platform GitHub, berkas bobot model disimpan secara publik pada tautan Google Drive berikut:

👉 **[KLIK DI SINI UNTUK MENGUNDUH FOLDER MODEL](https://drive.google.com/drive/folders/1pp0Zcxbqz1utJ2OtBW5zvrqDXv371-PA?usp=sharing)** 

Setelah diunduh, buat folder baru bernama `model_weights` di dalam direktori proyek ini, lalu ekstrak kelima berkas model tersebut ke dalamnya sehingga strukturnya menjadi seperti berikut:

```text
XLM-R-Indomie-Korean-Ramyeon-Series/
│
├── requirements.txt
├── inference.py
├── README.md
└── model_weights/               <-- Masukkan file model di sini
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── training_args
