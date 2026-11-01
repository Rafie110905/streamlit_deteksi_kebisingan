# Dokumentasi Implementasi: Server & Docker Deployment Readiness

- **Tanggal**: 31 Oktober 2026 (Diperbarui: 1 November 2026)
- **Modul**: Dockerfile, Streamlit Configuration, Runtime Dependencies, Server Hardening, Theme Styling
- **Proyek**: Deteksi Kebisingan Ruangan (`streamlit_deteksi_kebisingan`)

---

## 1. Ringkasan Masalah

Aplikasi ini disiapkan untuk dijalankan di lingkungan server menggunakan Docker. Beberapa kendala yang diidentifikasi dan diselesaikan:

1. **Path Direktori Konfigurasi Tidak Valid**:
   - `Dockerfile` mengeksekusi `COPY .streamlit/ .streamlit/`.
   - Sebelumnya berkas `config.toml` berada di *root directory*. Hal ini menyebabkan `docker build` gagal menemukan direktori `.streamlit`.
2. **Dependensi Runtime Tertinggal**:
   - Fitur "Deteksi realtime" mengimpor `from streamlit_webrtc import webrtc_streamer`.
   - Paket `streamlit-webrtc` belum tercantum pada `requirements.txt`.
3. **Aset Audio Peringatan Tidak Masuk Kontainer**:
   - `Dockerfile` sebelumnya tidak menyalin `peringatan_jangan_berisik.mp3`.
4. **Inkonsistensi Ekstensi Berkas Audio**:
   - Skrip Python mengacu ke `peringatan_jangan_berisik.mpeg`, sedangkan berkas fisik adalah `.mp3`.
5. **Standar Keamanan Server (Least Privilege)**:
   - Kontainer sebelumnya berjalan dengan hak akses `root`. Diperbarui agar berjalan di bawah user non-root `appuser` (UID 1000).
6. **Ketiadaan Healthcheck & `.dockerignore`**:
   - Ditambahkan `HEALTHCHECK` otomatis memantau endpoint `http://localhost:8501/_stcore/health`.
   - Dibuat `.dockerignore` agar context transfer Docker efisien.
7. **Inkonsistensi Tema Streamlit (Light vs Dark Mode)**:
   - Berkas `config.toml` lama memiliki `base = "light"` dan `backgroundColor = "#ffffff"`. Saat berkas ini dipindahkan ke lokasi resmi `.streamlit/config.toml`, tema putih tersebut aktif dan bertabrakan dengan kartu kustom serta visualisasi Matplotlib yang didesain untuk latar gelap (`#0f172a`).

---

## 2. Solusi yang Diterapkan

1. **Restrukturisasi Direktori Streamlit & Dark Mode Consistency**:
   - Folder `.streamlit/` dibuat dan diisi `config.toml`.
   - Tema disetel ke `base = "dark"` dengan warna latar `#090909`, panel `#171717`, teks `#f5f5f5`, dan aksen `#ef4444`.
   - Konfigurasi server standar Streamlit disertakan (`headless = true`, `address = "0.0.0.0"`, `port = 8501`).
2. **Pembaruan `requirements.txt`**:
   - Menambahkan `streamlit-webrtc`.
3. **Resolusi Cerdas Aset Audio (`app_deteksi_kebisingan.py`)**:
   - Menambahkan fungsi helper `cari_audio_peringatan()`.
4. **Hardening & Standarisasi `Dockerfile`**:
   - Memakai base image LTS stabil `python:3.11-slim-bookworm`.
   - Menambahkan utilitas `curl`, menyalin berkas audio mp3, membuat `appuser` (UID 1000), dan menambahkan `HEALTHCHECK`.
5. **Konfigurasi `docker-compose.yml`**:
   - Menetapkan nama kontainer `deteksi-kebisingan-app`, port mapping `8011:8501`, dan `restart: unless-stopped`.

---

## 3. Hasil Pengujian & Verifikasi

- **Engine**: Docker Desktop v29.9.2 (WSL 2 backend)
- **Container State**: **Up & Healthy** (`deteksi-kebisingan-app`)
- **Port Mapping**: `0.0.0.0:8011 -> 8501/tcp`
- **Uji Endpoint**:
  - `GET http://localhost:8011` -> `HTTP 200 OK`
  - `GET http://localhost:8011/_stcore/health` -> `HTTP 200 OK (ok)`
- **Tampilan**: Kembali ke **Dark Mode** yang konsisten dengan seluruh elemen UI dan grafik.
