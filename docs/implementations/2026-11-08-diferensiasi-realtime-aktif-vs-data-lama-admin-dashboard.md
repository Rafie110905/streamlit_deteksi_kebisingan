# Dokumentasi Implementasi: Diferensiasi Realtime Aktif (Live) vs Data Riwayat Lama pada Dashboard Admin

**Tanggal**: 8 November 2026  
**Status**: Selesai & Terverifikasi (100% Lulus Uji)  
**Komponen**: Logger SQLite (`deteksi_logger.py`), Streamlit Native Admin Panel (`app_deteksi_kebisingan.py`), FastApi Standalone Portal (`admin_panel/main.py`), Docker Runtime  

---

## 1. Latar Belakang & Masalah
Pada Dashboard Admin (baik antarmuka Native Streamlit maupun Standalone Portal WebSocket di Port 8000), kartu **STATUS AKUSTIK TERKINI** sebelumnya selalu menampilkan data baris terakhir dari tabel basis data tanpa penanda status keaktifan streaming.

Hal ini menimbulkan kebingungan bagi Administrator:
- Jika pengguna sebelumnya melakukan deteksi realtime lalu menutup browser atau menekan STOP, kartu admin tetap menampilkan label `📡 Realtime` dengan nilai VU meter terakhir (misalnya `-19 dBFS`).
- Admin tidak dapat membedakan: **"Apakah saat ini mikrofon pengguna sedang aktif menyiarkan suara realtime (ON / LIVE), ataukah ini hanya data lama dari sesi sebelumnya yang sudah terhenti (OFF / DATA LAMA)?"**

---

## 2. Solusi yang Diterapkan

Diterapkan mekanisme deteksi aliran audio real-time berbasis selisih waktu (*freshness timestamp delta*) dan penghitungan streamer aktif:

### A. Lapisan Database & Logger (`deteksi_logger.py`)
1. **Metadata Keaktifan pada `ambil_status_terkini()`**:
   - Menghitung selisih detik aktual (`detik_lalu = now - timestamp`).
   - Menyediakan format waktu ramah manusia (`keterangan_waktu`, contoh: "Baru saja", "14 detik lalu", "5 menit lalu").
   - Menentukan status aliran (*stream status*):
     - **`LIVE_AKTIF`**: Sumber audio adalah `Realtime WebRTC` dan pembaruan terakhir $\le 6$ detik. Nilai `is_live = True`.
     - **`REALTIME_TERHENTI`**: Sumber audio adalah `Realtime WebRTC` namun $\gt 6$ detik (klien telah berhenti merekam/menutup tab). Nilai `is_live = False`.
     - **`NON_REALTIME`**: Sumber audio berasal dari pengujian file upload atau rekaman manual statis.
2. **Statistik Streamer Aktif pada `hitung_statistik()`**:
   - Menghitung jumlah sesi klien yang aktif memancarkan audio dalam 6 detik terakhir:
     `SELECT COUNT(DISTINCT sesi_id) FROM deteksi WHERE sumber LIKE '%Realtime%' AND timestamp >= (now - 6s)`.
   - Mengembalikan parameter `"realtime_streaming_aktif": bool` dan `"total_streamer_aktif": int`.

### B. Antarmuka Native Streamlit (`app_deteksi_kebisingan.py`)
1. **Pembaruan Kartu Hero Status Admin (`_hero_status_html`)**:
   - **Saat Streaming Aktif**:
     Menampilkan pill badge:
     `🟢 LIVE STREAMING AKTIF (REALTIME ON)`
     Keterangan: `● Mikrofon Klien Sedang Siaran Langsung (Baru saja)`.
     Label VU Meter: `Digital VU Meter (Aliran Audio Live)`.
   - **Saat Terhenti / Data Lama**:
     Menampilkan pill badge:
     `🔴 REALTIME NONAKTIF (DATA LAMA / TERHENTI)`
     Keterangan: `⚠️ Mikrofon Klien Tidak Aktif — Menampilkan Data Rekaman Terakhir (N detik/menit lalu)`.
     Label VU Meter: `Digital VU Meter (Level Terakhir — Mikrofon Pengguna Saat Ini Nonaktif)`.
   - **Saat Pengujian Sampel File**:
     Menampilkan pill badge:
     `📁 DATA UJI SAMPEL (Upload / Rekam Langsung)`.
2. **Pembaruan Kartu KPI Sumber Audio**:
   - Kartu `📡 Realtime WebRTC` menyertakan sub-teks dinamis:
     `🟢 Klien Aktif Siaran (Live)` jika ada streamer aktif, atau `🔴 Tidak Ada Siaran Aktif (Idle)`.

### C. Portal Standalone FastAPI & WebSocket (`admin_panel/main.py`)
- Menambahkan elemen `#hero-live-badge` dan `#hero-status-note` pada template antarmuka.
- Fungsi JavaScript `applyLivePayload(d)` yang menerima data berkala via WebSocket/HTTP secara instan memperbarui:
  - Badge hijau menyala `🟢 LIVE STREAMING AKTIF (REALTIME ON)` saat klien bersuara.
  - Badge merah netral `🔴 REALTIME NONAKTIF (DATA LAMA)` saat klien berhenti.
  - Sub-teks pada kartu metrik realtime.

---

## 3. Berkas yang Diubah
- [deteksi_logger.py](file:///c:/Users/dedyn/Downloads/psd/psd%20tugas/streamlit_deteksi_kebisingan/deteksi_logger.py): Penambahan kalkulasi `detik_lalu`, `is_live`, `status_aliran`, dan `total_streamer_aktif`.
- [app_deteksi_kebisingan.py](file:///c:/Users/dedyn/Downloads/psd/psd%20tugas/streamlit_deteksi_kebisingan/app_deteksi_kebisingan.py): Pembaruan `_hero_status_html` dan kartu KPI di tab `Live Monitoring`.
- [admin_panel/main.py](file:///c:/Users/dedyn/Downloads/psd/psd%20tugas/streamlit_deteksi_kebisingan/admin_panel/main.py): Penambahan badge status live di HTML dan handler JavaScript WebSocket.

---

## 4. Hasil Verifikasi & Pengujian
1. **Validasi Respon API**:
   - Panggilan `GET /api/status?token=admin123` menghasilkan payload lengkap:
     ```json
     {
       "detik_lalu": 14,
       "keterangan_waktu": "14 detik lalu",
       "is_live": false,
       "status_aliran": "REALTIME_TERHENTI",
       "status_teks": "REALTIME NONAKTIF (DATA LAMA)",
       "realtime_streaming_aktif": false
     }
     ```
2. **Uji Keamanan & Integritas (Pentest Gate)**:
   - Akses tanpa token tetap ditolak `401 Unauthorized`.
   - Seluruh pengujian di `scratch/test_embed_and_tokens.py` lulus 100%.
3. **Deployment**:
   - Kedua kontainer Docker (`deteksi-kebisingan-1` dan `admin-panel-1`) telah disinkronkan dan beroperasi normal.
