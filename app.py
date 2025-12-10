# Import library yang dibutuhkan
from flask import Flask, render_template, request, jsonify
import sqlite3
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import random

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Nama file database SQLite
DB_NAME = 'travel.db'

# --- FUNGSI HELPER: KONEKSI DATABASE ---
def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row 
    return conn

# ============================================
# BAGIAN 1: HELPER (ANALISIS SENTIMEN) - Tidak Berubah
# ============================================
def analyze_sentiment_id(text):
    text = text.lower()
    pos_words = ['bagus', 'indah', 'bersih', 'nyaman', 'keren', 'puas', 'ramah', 'suka', 'enak', 'mantap', 'rekomendasi', 'luar biasa', 'senang', 'sejuk', 'strategis']
    neg_words = ['jelek', 'kotor', 'mahal', 'kecewa', 'buruk', 'kasar', 'macet', 'panas', 'bau', 'rusak', 'membosankan', 'rugi', 'parah']
    
    score = 0
    for word in pos_words:
        if word in text: score += 1
    for word in neg_words:
        if word in text: score -= 1
    
    if score > 0:
        return 0.6 + (min(score, 5) * 0.08)
    elif score < 0:
        return 0.4 - (min(abs(score), 5) * 0.08)
    else:
        return 0.5

def get_sentiment_label(score):
    if score >= 0.6: return 'Positif'
    if score <= 0.4: return 'Negatif'
    return 'Netral'

# ============================================
# BAGIAN 2: LOGIK REKOMENDASI SISTEM (HYBRID) - Tidak Berubah
# ============================================
def calculate_system_recommendation(places, user_prefs):
    scored_places = []
    for place in places:
        p = dict(place)
        place_cat = p['category']
        score_rating = p['rating'] / 5.0
        score_sentiment = p['sentiment_avg'] if p['sentiment_avg'] is not None else 0.5
        score_category = 0
        is_match = False
        for pref in user_prefs:
            if pref.lower() in place_cat.lower():
                is_match = True
                score_category = 0.33
                break
        if is_match:
            final_score = (score_rating * 0.33) + (score_sentiment * 0.34) + score_category
        else:
            final_score = (score_rating * 0.2) + (score_sentiment * 0.2)
        p['final_score'] = final_score
        p['match_percent'] = int(min(final_score * 100, 100))
        p['reco_type'] = 'system'
        scored_places.append(p)
    return sorted(scored_places, key=lambda x: x['final_score'], reverse=True)

# ============================================
# BAGIAN 3: LOGIK REKOMENDASI KOLABORATIF (USER-BASED KNN) - Tidak Berubah
# ============================================
def get_knn_collaborative_recommendation(conn, city, user_prefs, all_places_in_city, limit=2):
    print(f"\n--- Memulai USER-BASED KNN untuk Kota: {city}, Prefs: {user_prefs} ---")
    target_user_ratings = {}
    seed_place_ids = []
    for p in all_places_in_city:
        for pref in user_prefs:
            if pref.lower() in p['category'].lower():
                target_user_ratings[p['id']] = 5.0 
                seed_place_ids.append(p['id'])
                break
    
    if not target_user_ratings:
        if all_places_in_city:
             fallback_seeds = random.sample(all_places_in_city, min(3, len(all_places_in_city)))
             for p in fallback_seeds:
                 target_user_ratings[p['id']] = 5.0
                 seed_place_ids.append(p['id'])
        else:
             return []

    query = "SELECT user_id, place_id, rating_given FROM reviews WHERE user_id IS NOT NULL"
    reviews_df = pd.read_sql_query(query, conn)
    if reviews_df.empty: return []

    user_item_matrix = reviews_df.pivot_table(index='user_id', columns='place_id', values='rating_given').fillna(0)
    all_places_db = conn.execute("SELECT id FROM places").fetchall()
    all_place_ids_db = [r['id'] for r in all_places_db]
    user_item_matrix = user_item_matrix.reindex(columns=all_place_ids_db, fill_value=0)
    user_item_matrix_sparse = csr_matrix(user_item_matrix.values)

    try:
        model_knn = NearestNeighbors(n_neighbors=4, algorithm='brute', metric='cosine')
        model_knn.fit(user_item_matrix_sparse)
    except Exception as e:
        print(f"Error fitting KNN model: {e}")
        return []

    target_user_vector = pd.Series(0.0, index=user_item_matrix.columns)
    for pid, rating in target_user_ratings.items():
        if pid in target_user_vector.index:
             target_user_vector[pid] = rating
            
    distances, indices = model_knn.kneighbors(target_user_vector.values.reshape(1, -1))
    similar_user_ids = user_item_matrix.index[indices.flatten()].tolist()

    similar_users_ratings = reviews_df[reviews_df['user_id'].isin(similar_user_ids)]
    well_rated_places = similar_users_ratings[similar_users_ratings['rating_given'] >= 4]
    city_place_ids = set([p['id'] for p in all_places_in_city])
    reco_candidates = well_rated_places[well_rated_places['place_id'].isin(city_place_ids)]
    reco_candidates = reco_candidates[~reco_candidates['place_id'].isin(seed_place_ids)]

    if reco_candidates.empty:
        return []

    reco_summary = reco_candidates.groupby('place_id').agg(
        avg_rating=('rating_given', 'mean'),
        voter_count=('user_id', 'nunique')
    ).reset_index()
    
    reco_summary = reco_summary.sort_values(by=['voter_count', 'avg_rating'], ascending=[False, False]).head(limit)
    final_place_ids = reco_summary['place_id'].tolist()
    
    collaborative_results = []
    for p_raw in all_places_in_city:
        if p_raw['id'] in final_place_ids:
            p = dict(p_raw)
            stats = reco_summary[reco_summary['place_id'] == p['id']].iloc[0]
            p['match_percent'] = int((stats['avg_rating'] / 5.0) * 100)
            p['reco_type'] = 'collaborative'
            voter_count_int = int(stats['voter_count']) 
            p['collab_info'] = f"Disukai oleh {voter_count_int} wisatawan dengan selera mirip Anda"
            p['voter_count_raw'] = voter_count_int
            collaborative_results.append(p)

    collaborative_results.sort(key=lambda x: x['voter_count_raw'], reverse=True)
    return collaborative_results

# ============================================
# BAGIAN 4: LOGIK REKOMENDASI GNN (SIMULASI GRAF)
# ============================================
def get_gnn_simulated_recommendation(all_places_in_city, user_prefs, seen_ids, limit=1):
    print(f"\n--- Memulai Simulasi GNN untuk Prefs: {user_prefs} ---")
    # Konsep Graf: Mencari "Node Tetangga" yang kuat.
    # Dalam simulasi ini, "tetangga kuat" adalah tempat dengan KATEGORI SAMA PERSIS
    # dan memiliki rating tertinggi di kota tersebut.

    # 1. Cari kategori utama yang diminati user
    target_category = None
    for p in all_places_in_city:
        for pref in user_prefs:
            if pref.lower() in p['category'].lower():
                target_category = p['category'] # Ambil kategori lengkapnya
                break
        if target_category: break
    
    if not target_category and user_prefs:
         # Fallback jika tidak ada match persis, ambil pref pertama
         target_category = user_prefs[0]

    if not target_category: return []

    print(f"Target Kategori untuk Graf: {target_category}")

    # 2. Cari "Node Pemicu" (Tempat dengan rating tertinggi di kategori tersebut)
    # Ini akan jadi jangkar penjelasan grafnya nanti.
    category_places = [p for p in all_places_in_city if target_category.lower() in p['category'].lower()]
    category_places.sort(key=lambda x: x['rating'], reverse=True)
    
    if not category_places: return []
    
    top_anchor_place = category_places[0] # Tempat terbaik di kategori ini

    # 3. Cari Rekomendasi GNN (Tempat lain di kategori sama, rating bagus, belum dilihat)
    gnn_candidates = []
    for p in category_places:
        if p['id'] != top_anchor_place['id'] and p['id'] not in seen_ids and p['rating'] >= 4.0:
            cand = dict(p)
            cand['match_percent'] = int((p['rating'] / 5.0) * 100)
            cand['reco_type'] = 'gnn'
            # Simpan info anchor untuk visualisasi nanti
            cand['anchor_name'] = top_anchor_place['name']
            cand['anchor_category'] = target_category
            cand['collab_info'] = f"Rekomendasi berbasis Graf: Terhubung kuat dengan {top_anchor_place['name']}"
            gnn_candidates.append(cand)
    
    # Ambil top N dari kandidat
    return gnn_candidates[:limit]

# ============================================
# BAGIAN 5: ROUTES (Endpoint API & Halaman Web)
# ============================================

@app.route('/')
def index():
    return render_template('index.html')

# ROUTE BARU UNTUK VISUALISASI GRAF
@app.route('/graph_visualization/<int:place_id>')
def graph_visualization(place_id):
    conn = get_db_connection()
    
    # Ambil data tempat yang direkomendasikan
    target_place = conn.execute("SELECT * FROM places WHERE id = ?", (place_id,)).fetchone()
    
    if not target_place:
        conn.close()
        return "Tempat tidak ditemukan", 404
        
    # Untuk simulasi visualisasi, kita perlu mencari "Anchor Place" (tempat populer yg mirip)
    # Kita cari tempat lain di kota yang sama dengan kategori yang sama dan rating tertinggi
    query_anchor = """
        SELECT * FROM places 
        WHERE city = ? AND category = ? AND id != ?
        ORDER BY rating DESC LIMIT 1
    """
    anchor_place = conn.execute(query_anchor, (target_place['city'], target_place['category'], place_id)).fetchone()
    
    conn.close()
    
    # Siapkan data untuk dikirim ke template
    graph_data = {
        'target': dict(target_place),
        'anchor': dict(anchor_place) if anchor_place else None,
        'category': target_place['category']
    }
    
    return render_template('graph_visual.html', data=graph_data)


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    city = data.get('city')
    budget = int(data.get('budget'))
    prefs = data.get('preferences', [])
    
    print(f"\nRequest diterima: Kota={city}, Budget={budget}, Prefs={prefs}")

    conn = get_db_connection()
    
    query = "SELECT * FROM places WHERE city = ? AND price <= ?"
    places_raw_sqlite = conn.execute(query, (city, budget)).fetchall()
    places_raw = [dict(row) for row in places_raw_sqlite]

    if not places_raw:
        conn.close()
        return jsonify([])

    # Setup Jumlah Rekomendasi
    N_SYSTEM_TOP = 2
    N_COLLAB_TOP = 2 # Diubah jadi 1
    N_GNN_TOP = 1    # Ditambah 1 untuk GNN

    # 1. Hitung Sistem & Collab
    system_recos_all = calculate_system_recommendation(places_raw, prefs)
    collab_recos = get_knn_collaborative_recommendation(conn, city, prefs, places_raw, limit=N_COLLAB_TOP)

    final_results = []
    seen_ids = set()

    # A. Masukkan Top N Sistem
    for i in range(min(len(system_recos_all), N_SYSTEM_TOP)):
        reco = system_recos_all[i]
        final_results.append(reco)
        seen_ids.add(reco['id'])
        
    # B. Masukkan Top N Collab
    added_collab = 0
    for reco in collab_recos:
        if reco['id'] not in seen_ids and added_collab < N_COLLAB_TOP:
            final_results.append(reco)
            seen_ids.add(reco['id'])
            added_collab += 1

    # 2. Hitung GNN (Setelah tahu apa yang sudah direkomendasikan agar tidak duplikat)
    # Kita kirim 'seen_ids' agar GNN mencari tempat baru
    gnn_recos = get_gnn_simulated_recommendation(places_raw, prefs, seen_ids, limit=N_GNN_TOP)

    # C. Masukkan Top N GNN
    added_gnn = 0
    for reco in gnn_recos:
         # Double check seen_ids meskipun sudah difilter di fungsi
        if reco['id'] not in seen_ids and added_gnn < N_GNN_TOP:
            final_results.append(reco)
            seen_ids.add(reco['id'])
            added_gnn += 1

    conn.close()
    print(f"Total rekomendasi dikirim: {len(final_results)}")
    return jsonify(final_results)

@app.route('/submit_review', methods=['POST'])
def submit_review():
    # (Tidak ada perubahan)
    data = request.json
    place_id = data['place_id']
    comment = data['comment']
    rating = int(data['rating'])
    score = analyze_sentiment_id(comment)
    label = get_sentiment_label(score)
    conn = get_db_connection()
    conn.execute("INSERT INTO reviews (place_id, comment, sentiment_score, sentiment_label, rating_given) VALUES (?, ?, ?, ?, ?)",
                 (place_id, comment, score, label, rating))
    cur = conn.cursor()
    new_avg = cur.execute("SELECT AVG(sentiment_score) FROM reviews WHERE place_id = ?", (place_id,)).fetchone()[0]
    conn.execute("UPDATE places SET sentiment_avg = ? WHERE id = ?", (new_avg, place_id))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')