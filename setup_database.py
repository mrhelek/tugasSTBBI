import sqlite3
import pandas as pd
import random
import os

# --- Konfigurasi ---
DB_NAME = 'travel.db'
CSV_PATH = 'data/tourism_with_id.csv' # Pastikan file ini ada

# --- Analisis Sentimen Sederhana ---
def analyze_sentiment_id(text):
    text = text.lower()
    pos_words = ['bagus', 'indah', 'bersih', 'nyaman', 'keren', 'puas', 'ramah', 'suka', 'enak', 'mantap', 'rekomendasi', 'luar biasa', 'senang', 'sejuk', 'strategis']
    neg_words = ['jelek', 'kotor', 'mahal', 'kecewa', 'buruk', 'kasar', 'macet', 'panas', 'bau', 'rusak', 'membosankan', 'rugi', 'parah']
    
    score = 0
    for word in pos_words:
        if word in text: score += 1
    for word in neg_words:
        if word in text: score -= 1
    
    if score > 0: return 0.6 + (min(score, 5) * 0.08)
    elif score < 0: return 0.4 - (min(abs(score), 5) * 0.08)
    else: return 0.5

def get_sentiment_label(score):
    if score >= 0.6: return 'Positif'
    if score <= 0.4: return 'Negatif'
    return 'Netral'

# --- Template Review Dummy ---
dummy_comments = {
    5: ["Tempat yang luar biasa!", "Sangat puas berkunjung ke sini.", "Pemandangan indah dan fasilitas lengkap.", "Wajib dikunjungi, sangat berkesan.", "Liburan terbaik di sini."],
    4: ["Tempatnya bagus, cukup nyaman.", "Pengalaman yang menyenangkan.", "Lumayan untuk liburan keluarga.", "Fasilitas oke, tapi agak ramai.", "Bagus untuk foto-foto."],
    3: ["Biasa saja, standar.", "Cukup oke, tapi tidak ada yang spesial.", "Not bad, tapi antriannya panjang.", "Lumayan untuk sekedar mampir."],
    2: ["Kurang memuaskan.", "Tempatnya agak kotor dan tidak terawat.", "Harga terlalu mahal untuk fasilitasnya.", "Akses jalan susah."],
    1: ["Sangat mengecewakan.", "Pelayanan buruk sekali.", "Tidak sesuai ekspektasi, rugi waktu.", "Tidak akan kembali lagi."]
}

def init_db():
    # Hapus database lama agar bersih
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
        print("Database lama dihapus. Membuat database baru...")
        
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # 1. Tabel Places
    c.execute('''CREATE TABLE places 
                 (id INTEGER PRIMARY KEY, name TEXT, category TEXT, city TEXT, 
                  price INTEGER, rating REAL, image_url TEXT, sentiment_avg REAL)''')

    # 2. Tabel Reviews (User History Matrix)
    c.execute('''CREATE TABLE reviews 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, place_id INTEGER, user_id INTEGER,
                  comment TEXT, sentiment_score REAL, sentiment_label TEXT, rating_given INTEGER)''')

    print("Membaca CSV dan memproses data tempat...")
    try:
        df = pd.read_csv(CSV_PATH)
        all_place_ids = []
        for index, row in df.iterrows():
            p_id = row['Place_Id']
            all_place_ids.append(p_id)
            # Gunakan layanan gambar placeholder
            image_url = f"https://dummyimage.com/600x400/008080/ffffff&text={row['Place_Name'].replace(' ', '+')}"
            # Set sentiment awal netral (0.5), nanti diupdate
            c.execute("INSERT INTO places VALUES (?, ?, ?, ?, ?, ?, ?, ?)", 
                      (p_id, row['Place_Name'], row['Category'], row['City'], row['Price'], row['Rating'], image_url, 0.5))
    except FileNotFoundError:
        print(f"ERROR: File {CSV_PATH} tidak ditemukan.")
        return

    # --- PERUBAHAN DATA DUMMY DISINI ---
    print("Men-generate DUMMY USER HISTORY (Collaborative Data) yang LEBIH BANYAK...")
    
    # Tingkatkan jumlah user secara signifikan
    num_dummy_users = 300  # Sebelumnya 50
    total_reviews_generated = 0
    
    print(f"Membuat riwayat untuk {num_dummy_users} user dummy. Mohon tunggu...")

    for user_id in range(1, num_dummy_users + 1):
        # Tingkatkan jumlah tempat yang dikunjungi per user agar overlap lebih banyak
        num_visited = random.randint(10, 25) # Sebelumnya 5-15
        
        # Pastikan tidak error jika jumlah tempat di CSV sedikit
        if num_visited > len(all_place_ids):
            num_visited = len(all_place_ids)
            
        visited_places = random.sample(all_place_ids, num_visited)
        
        for p_id in visited_places:
            # Beri bobot rating. Cenderung positif, tapi tetap ada variasi.
            # Weights menentukan probabilitas munculnya rating 5,4,3,2,1
            rating = random.choices([5, 4, 3, 2, 1], weights=[0.35, 0.35, 0.15, 0.1, 0.05])[0]
            
            comment = random.choice(dummy_comments[rating])
            score = analyze_sentiment_id(comment)
            label = get_sentiment_label(score)
            
            # user_id DIISI untuk data kolaboratif
            c.execute("INSERT INTO reviews (place_id, user_id, comment, sentiment_score, sentiment_label, rating_given) VALUES (?, ?, ?, ?, ?, ?)",
                      (p_id, user_id, comment, score, label, rating))
            total_reviews_generated += 1
            
        # Print progres setiap 50 user
        if user_id % 50 == 0:
            print(f"...Progres: {user_id} user diproses.")

    print(f"SELESAI. Berhasil membuat {total_reviews_generated} data riwayat review.")

    # Update rata-rata sentimen di tabel places
    print("Mengupdate rata-rata sentimen tempat...")
    for p_id in all_place_ids:
        avg_sent = c.execute("SELECT AVG(sentiment_score) FROM reviews WHERE place_id = ?", (p_id,)).fetchone()[0]
        if avg_sent is not None:
            c.execute("UPDATE places SET sentiment_avg = ? WHERE id = ?", (avg_sent, p_id))

    conn.commit()
    conn.close()
    print(f"Database '{DB_NAME}' siap digunakan dengan data yang KAYA!")

if __name__ == '__main__':
    init_db()