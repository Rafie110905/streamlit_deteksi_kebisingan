# Dashboard Admin Multi-Menu & Realtime Streaming (Smart Noise Intelligence)

**Tanggal**: 8 November 2026  
**Status**: Selesai & Terverifikasi (100% Lulus Uji Penetrasi & Integrasi)  
**Modul Terdampak**: `admin_panel/main.py`, `deteksi_logger.py`, Docker Service `admin-panel`  

---

## 1. Latar Belakang & Kebutuhan Fitur

Pengguna membutuhkan antarmuka dashboard admin yang lebih kaya fitur, memiliki navigasi multi-menu yang terstruktur, serta menyajikan pembaruan data secara *realtime* tanpa perlu me-refresh halaman secara manual.

Fitur-fitur utama yang ditambahkan:
1. **Navigasi Multi-Menu (Sidebar)**:
   - **📡 Monitoring Realtime**: Status vonis instan, Digital VU Meter (Sound Level Bar), grafik aliran level suara 10 menit, dan alarm buzzer audio.
   - **📊 Analisis & Statistik Mendalam**: Pola kebisingan berdasarkan jam (24 jam), komparasi akurasi/vonis per metode (Random Forest vs KNN vs SVM vs RMS), distribusi tingkat keyakinan, dan ringkasan nilai ekstrem (*peak*, *average*, *floor* dBFS).
   - **📋 Riwayat & Audit Log**: Log deteksi interaktif dengan filter vonis, filter algoritma, live debounce search (pencarian nama file/sesi), serta tombol ekspor data ke file CSV.
   - **👥 Sesi Pengguna Streamlit**: Monitoring pengguna aktif dengan status *Online* (<60s), *Idle*, atau *Offline*, serta tombol pembersihan sesi tidak aktif.
   - **⚙️ Sistem & Model**: Status kesehatan database SQLite, ukuran file, total baris records, metrik akurasi model ML aktif, dan referensi standar regulasi kebisingan (Permenaker No. 5/2018).
2. **Konektivitas Realtime WebSocket & Auto-Sync Fallback**:
   - Kanal WebSocket (`/ws/live`) berkecepatan tinggi yang mengirimkan pembaruan data setiap 1 detik.
   - Mekanisme *auto-reconnect* dan *HTTP polling fallback* otomatis jika koneksi WebSocket terputus.

---

## 2. Arsitektur & Pemisahan Tanggung Jawab (Separation of Concerns)

Sesuai prinsip arsitektur modular:
1. **Lapisan Database (`deteksi_logger.py`)**:
   - Menambahkan fungsi agregasi independen: `ambil_riwayat_terfilter()`, `ambil_analisis_mendalam()`, `ambil_daftar_sesi()`, `bersihkan_sesi_manual()`, dan `ambil_status_sistem()`.
   - Menjaga isolasi thread-safe menggunakan threading lock (`_lock`) saat membaca dan menulis database SQLite.
2. **Lapisan Presentasi & API (`admin_panel/main.py`)**:
   - Berjalan pada FastAPI dan Uvicorn di port `8000`.
   - Menggunakan routing RESTful terstruktur (`/api/status`, `/api/riwayat`, `/api/grafik`, `/api/analisis`, `/api/sesi`, `/api/sistem`, `/api/ekspor/csv`) dan WebSocket (`/ws/live`).
   - Tampilan antarmuka *single-page application* (SPA) dengan tab navigation instan berbasis CSS/JS native tanpa re-render halaman.

---

## 3. Standar Keamanan & Hasil Uji Penetrasi (Pen-Test Gate)

Sesuai aturan keamanan global, seluruh endpoint mutasi dan pembacaan data diuji secara kritis terhadap ancaman *unauthorized access*:

1. **Bypass Test (Tanpa Kredensial)**:
   - `GET /api/status` $\rightarrow$ **401 Unauthorized** (LULUS)
   - `GET /api/analisis` $\rightarrow$ **401 Unauthorized** (LULUS)
   - `GET /api/sesi` $\rightarrow$ **401 Unauthorized** (LULUS)
   - `POST /api/sesi/bersihkan` $\rightarrow$ **401 Unauthorized** (LULUS)
2. **Authenticated Access Test (`admin:admin123`)**:
   - `GET /` $\rightarrow$ **200 OK** (HTML Dashboard Multi-Menu)
   - `GET /api/status` $\rightarrow$ **200 OK**
   - `GET /api/riwayat` $\rightarrow$ **200 OK**
   - `GET /api/grafik` $\rightarrow$ **200 OK**
   - `GET /api/analisis` $\rightarrow$ **200 OK**
   - `GET /api/sesi` $\rightarrow$ **200 OK**
   - `GET /api/sistem` $\rightarrow$ **200 OK**
   - `GET /api/ekspor/csv` $\rightarrow$ **200 OK** (File download streaming)
   - `POST /api/sesi/bersihkan` $\rightarrow$ **200 OK** (Mutasi berhasil diproteksi)
3. **WebSocket Realtime Stream Test (`/ws/live`)**:
   - Berhasil terkoneksi dan menerima paket payload data setiap 1 detik secara konsisten.

---

## 4. Cara Akses & Kredensial Admin

- **URL Dashboard Admin**: `http://localhost:8000/`
- **Username**: `admin`
- **Password**: `admin123`
