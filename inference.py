import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =====================================================================
# 1. KAMUS KATA KUNCI ASPEK MULTILINGUAL (LEXICON)
# =====================================================================
aspect_lexicon = {
    'Promosi': [
        'jakarta', 'iklan', 'konser', 'promosi', 'event luring', 'booth', 'musik', 'krl',
        'pop store', 'pop-up', 'pop up', 'popstore', 'newjeans pop', 'ohmygooditsindomie', 'newjeans ad',
        'oh good', 'ad campaign', 'music', 'promotion', 'commercial', 'benefit', 'event', 'concert',
        'billboard', 'takeover jakarta', 'ads', 'ad',
        '인스타', '스토리', '인스타그램', '인스타 릴스', '릴스 뉴진스', '릴스', '콜라보'
    ],
    'Rasa': [
        'enak', 'lezat', 'mantap', 'mantul', 'pedas', 'pedes', 'asin', 'micin', 'lidah',
        'gurih', 'bumbu',  'manis', 'asam', 'sedap', 'top', 'juara', 'nagih', 'doyan',
        'tekstur', 'kenyal', 'chewy', 'lembut', 'aroma', 'wangi',  'segar', 'seger',
        'kuah', 'goreng', 'varian', 'rasa', 'rasanya', 'pedasnya', 'pedesnya',
        'k-rose', 'spicy ramyeon', 'fiery chikin', 'biru', 'pink', 'merah',
        'delicious', 'yummy', 'spicy', 'hot', 'seasoning', 'addictive', 'tasty',
        'taste', 'flavor', 'flavour', 'creamy',
        '라면', '한국라면', '미고랭', '매운 맛', '맛', '맛있어요'
    ],
    'Kolaborasi': [
        'newjeans', 'hanni', 'haerin', 'minji', 'danielle', 'hyein', 'idol', 'kpop',
        'girlgroup', 'ambassador', 'ba', 'brandambassador', 'kolaborasi', 'collab',
        'collaboration', 'bunny', 'bunnies', 'fandom', 'fans', 'fanbase', 'bias',
        'stan', 'stanning', 'official', 'photocard', 'pc', 'postcard', 'merch',
        'merchandise', 'poster', 'sticker', 'limited', 'edition', 'spesial', 'special',
        'global brand', 'brand ambassador', 'team bunnies', 'njz', 'nj', 'new jeans',
        '뉴진스', '모델', '포카', '뉴진스가 광고하는'
    ]
}

def deteksi_aspek_otomatis(teks):
    """Mendeteksi kategori aspek berdasarkan kecocokan kata kunci dalam teks ulasan"""
    teks_lower = teks.lower()
    skor_aspek = {kategori: 0 for kategori in aspect_lexicon.keys()}
    
    # Hitung jumlah kata kunci yang cocok untuk setiap aspek
    for kategori, keywords in aspect_lexicon.items():
        for kw in keywords:
            if kw in teks_lower:
                skor_aspek[kategori] += 1
                
    # Ambil aspek dengan skor kecocokan tertinggi
    aspek_terpilih = max(skor_aspek, key=skor_aspek.get)
    
    # Jika tidak ada kata kunci yang cocok sama sekali, default ke aspek terpopuler (Rasa)
    if skor_aspek[aspek_terpilih] == 0:
        return 'Rasa'
    return aspek_terpilih

# =====================================================================
# 2. PROSES INISIALISASI MODEL & TOKENIZER SECARA LOKAL
# =====================================================================
# Menunjuk ke folder lokal tempat meletakkan file model unduhan dari GDrive
MODEL_DIR = "./model_weights"

if not os.path.exists(MODEL_DIR):
    print(f"❌ Error: Folder '{MODEL_DIR}' tidak ditemukan!")
    print("Silakan buat folder tersebut dan masukkan 5 berkas model hasil unduhan dari Google Drive.")
    sys.exit(1)

print("⏳ Sedang memuat model XLM-RoBERTa dari folder lokal...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    print("✅ Model dan Tokenizer berhasil dimuat ke memori!")
except Exception as e:
    print(f"❌ Gagal memuat model. Pastikan semua file lengkap di dalam folder '{MODEL_DIR}'.")
    print(f"Detail error: {e}")
    sys.exit(1)

# =====================================================================
# 3. ANTARMUKA PENGUJIAN INTERAKTIF (CLI)
# =====================================================================
id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}

print("\n" + "="*60)
print("  APLIKASI ANALISIS SENTIMEN BERBASIS ASPEK (ABSA) OTOMATIS  ")
print("               INDOMIE KOREAN RAMYEON SERIES                 ")
print("="*60)
print("Petunjuk: Masukkan kalimat ulasan Anda (bisa Indonesia, Inggris, atau Korea).")
print("Ketik 'exit' untuk keluar dari program.\n")

while True:
    input_user = input("✍️ Masukkan ulasan: ")
    if input_user.strip().lower() == 'exit':
        print("\nTerima kasih! Sampai jumpa di sidang skripsi.")
        break
        
    if not input_user.strip():
        continue
        
    # A. Deteksi aspek secara otomatis menggunakan lexicon
    aspek_terdeteksi = deteksi_aspek_otomatis(input_user)
    
    # B. Susun format input yang dikenali oleh XLM-RoBERTa
    formatted_input = f"Aspect: {aspek_terdeteksi} | Text: {input_user}"
    
    # C. Jalankan proses prediksi sentimen
    inputs = tokenizer(formatted_input, padding=True, truncation=True, max_length=160, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        
    # D. Tampilkan hasil prediksi
    print("-" * 60)
    print(f"🔎 Aspek Terdeteksi : {aspek_terdeteksi} (Otomatis)")
    print(f"🎭 Prediksi Sentimen: {id2label[pred_idx]}")
    print(f"🎯 Tingkat Akurasi  : {probs[0][pred_idx].item()*100:.2f}%")
    print("-" * 60 + "\n")
