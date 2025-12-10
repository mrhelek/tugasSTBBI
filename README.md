berikut adalah struktur folder dan file untuk menjalakan aplikasi berbasis web dengan kode python, yang pertama dilakukan adalah deploy database dengan menjalakan setup_database.py kemudian baru menjalakan aplikasi app.py

📁 smart_travel_app/          <-- Folder Utama
│
├── 📄 app.py                 <-- File utama aplikasi Flask 
├── 📄 setup_database.py      <-- Script untuk membuat database & data dummy
├── 🛢️ travel.db              <-- Database SQLite 
│
├── 📂 data/                  <-- Folder untuk menyimpan dataset mentah
│   └── 📄 tourism_with_id.csv  <-- File CSV dataset pariwisata 
│
└── 📂 templates/             <-- Folder KHUSUS untuk file HTML
    ├── 📄 index.html         <-- Halaman utama dashboard rekomendasi
    └── 📄 graph_visual.html  <-- Halaman visualisasi graf 
