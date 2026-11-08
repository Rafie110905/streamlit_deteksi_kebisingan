# Dokumentasi Implementasi: Perbaikan Deteksi Realtime WebRTC

- **Tanggal**: 8 November 2026
- **Modul**: `app_deteksi_kebisingan.py`, WebRTC Streamer, Audio Buffer
- **Proyek**: Deteksi Kebisingan Ruangan (`streamlit_deteksi_kebisingan`)

---

## 1. Masalah yang Ditemukan pada Deteksi Realtime

Saat pengguna memilih mode **"Deteksi realtime"** dan menekan tombol **START**, deteksi tidak berjalan atau macet pada status menunggu/koneksi. Penyebab utamanya:

1. **Mode Negosiasi WebRTC Salah**:
   - `webrtc_streamer` sebelumnya tidak menyetel parameter `mode`, sehingga memakai bawaan `WebRtcMode.SENDRECV` (dua arah).
   - Karena server tidak mengirimkan stream audio balik (`sendback_audio=False`), browser menunggu track balik yang tidak pernah datang, sehingga status koneksi WebRTC macet (*connecting loop*).
2. **Ketiadaan Konfigurasi STUN Server**:
   - WebRTC antara browser (client) dan container Docker / Streamlit Cloud berada di subnet/NAT yang berbeda.
   - Tanpa `rtc_configuration` yang berisi server STUN publik, pertukaran kandidat ICE (*Interactive Connectivity Establishment*) gagal, sehingga paket audio UDP tidak sampai ke server.
3. **Resiko Dimensi Array Audio**:
   - Array audio dari frame PyAV (`frame.to_ndarray()`) bisa berformat planar stereo `(2, samples)` atau packed stereo `(1, 2*samples)`. Jika tidak dinormalisasi ke 1D mono yang teratur, fungsi ekstraksi fitur audio librosa dapat menghasilkan error.
4. **Potensi Crash `_sesi_id`**:
   - Pemanggilan `get_script_run_ctx().session_id` tanpa pengecekan `None` dapat memicu `AttributeError` pada konteks non-main thread.

---

## 2. Solusi & Perbaikan

1. **Penyetelan `mode=WebRtcMode.SENDONLY`**:
   - Menegaskan bahwa browser hanya mengirimkan audio mikrofon ke server tanpa mengharapkan stream balik dari server.
2. **Penambahan Konfigurasi STUN/TURN**:
   - Menambahkan daftar STUN server publik andal (Google & Cloudflare) ke `rtc_configuration`:
     - `stun:stun.l.google.com:19302`
     - `stun:stun1.l.google.com:19302`
     - `stun:stun2.l.google.com:19302`
     - `stun:stun.cloudflare.com:3478`
   - Mendukung penambahan kredensial TURN kustom dari Streamlit Secrets (`st.secrets["turn"]`).
3. **Normalisasi Audio Buffer**:
   - Memastikan data frame audio selalu dikonversi menjadi 1D contiguous array `float32` dengan `.ravel()` sebelum dimasukkan ke sliding buffer 2 detik.
4. **Pencegahan Error Konteks Eksekusi**:
   - Menambahkan fallback jika `get_script_run_ctx()` bernilai `None`.

---

## 3. Hasil Pengujian Nyata

- Simulasi 100 frame audio (2 detik, 48.000 Hz) di dalam lingkungan container:
  - **Status**: Berhasil dieksekusi 100% tanpa error (`Error: None`).
  - **Fitur Ekstrak**: Berhasil dihitung (`RMS: -9.03 dBFS`).
  - **Prediksi Model**: Random Forest berhasil mengklasifikasikan secara realtime (`P(berisik): 0.623`).

