# Deteksi Kebisingan Ruangan

Aplikasi Streamlit untuk menganalisis rekaman audio dan memperkirakan apakah suara ruangan **berisik** atau **tidak berisik**. Audio bisa diunggah atau direkam langsung melalui mikrofon browser.

Aplikasi daring: https://psd-deteksi-kebisingan.streamlit.app

## Fitur

- Menerima audio `.mp3`, `.wav`, `.m4a`, `.flac`, dan `.ogg`.
- Merekam audio langsung dari mikrofon melalui browser.
- Mendeteksi suara secara realtime saat mikrofon aktif, dengan prediksi baru setiap detik dari jendela audio 2 detik.
- Menghitung fitur audio seperti RMS, zero-crossing rate, spectral centroid, rolloff, bandwidth, dan flatness.
- Menyediakan pilihan metode deteksi yang terpisah: Random Forest atau RMS.
- Random Forest memakai 300 pohon dan menampilkan probabilitas model, voting pohon, serta grafik analisis audio.
- Metode RMS membandingkan level audio dalam dBFS dengan ambang yang dapat diatur; metode ini tidak memakai prediksi Random Forest.
- Memutar audio `peringatan_jangan_berisik.mpeg` saat hasil Rekam langsung atau Deteksi realtime menyatakan **BERISIK**, pada kedua metode.
- Menyimpan hasil pengujian selama sesi aplikasi masih berjalan.

## Struktur berkas

```text
.
├── Dockerfile
├── docker-compose.yml
├── app_deteksi_kebisingan.py
├── fitur_audio_kebisingan.csv
├── peringatan_jangan_berisik.mpeg
├── packages.txt
└── requirements.txt
```

`fitur_audio_kebisingan.csv` berisi fitur dan label data latih yang diperlukan aplikasi. Aplikasi tidak melatih model dari audio yang diunggah; model dilatih saat aplikasi dimulai menggunakan data fitur tersebut.

## Persyaratan

- Python dengan paket pada `requirements.txt`.
- Untuk perekaman langsung, browser harus memiliki akses ke mikrofon. Jika diminta, izinkan akses mikrofon untuk situs aplikasi.
- Deteksi realtime memerlukan browser dengan dukungan WebRTC. Di luar localhost, halaman aplikasi harus disajikan melalui HTTPS; jaringan deployment mungkin juga memerlukan konfigurasi STUN/TURN.
- `packages.txt` mencantumkan paket sistem Linux (`ffmpeg` dan `libsndfile1`) untuk lingkungan deployment yang mendukung berkas tersebut.

## Menjalankan secara lokal

Dari folder proyek, buat dan aktifkan virtual environment, lalu pasang dependensi:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Jalankan aplikasi:

```powershell
python -m streamlit run app_deteksi_kebisingan.py
```

Streamlit akan menampilkan alamat aplikasi di terminal dan membukanya di browser.

## Menjalankan dengan Docker

Pastikan Docker Engine dan Docker Compose tersedia, lalu jalankan perintah berikut dari folder proyek:

```powershell
docker compose up --build
```

Compose akan membangun image dari `Dockerfile` dan menjalankan aplikasi. Buka [http://localhost:8011](http://localhost:8011). Port host `8011` digunakan untuk kelompok 11 dan diteruskan ke port Streamlit `8501` di dalam container.

Untuk menghentikan aplikasi, tekan `Ctrl+C`, lalu jalankan:

```powershell
docker compose down
```

## Menguji audio

1. Buka tab **Uji Sampel**.
2. Pilih **Random Forest** atau **RMS** sebagai metode deteksi.
3. Jika memilih RMS, atur **Ambang suara keras (RMS, dBFS)** bila diperlukan.
4. Pilih **Unggah berkas** dan pilih audio, atau pilih **Rekam langsung**.
5. Untuk rekaman langsung, izinkan akses mikrofon pada browser dan lakukan rekaman.
6. Analisis berjalan otomatis setelah berkas dipilih atau rekaman selesai. Hasil dan grafik akan muncul tanpa menekan tombol. Jika menggunakan **Rekam langsung** dan hasilnya **BERISIK**, aplikasi juga memutar suara peringatan.

Untuk deteksi berkelanjutan, pilih metode deteksi, lalu pilih **Deteksi realtime**, tekan **START**, dan izinkan akses mikrofon. Hasil pertama muncul setelah 2 detik audio dan diperbarui setiap detik selama sesi WebRTC aktif. Jika hasil berubah menjadi **BERISIK**, aplikasi memutar satu kali suara peringatan; bunyi berikutnya baru diputar setelah status kembali tenang lalu berisik lagi. Tekan **STOP** untuk mengakhiri deteksi.

Jika browser memblokir pemutaran otomatis, tekan tombol putar pada pemutar peringatan yang ditampilkan.

Mengubah ambang RMS akan memperbarui vonis dari hasil analisis yang sama, tanpa memproses audio ulang. Ambang hanya digunakan saat metode **RMS** dipilih. Hasil dari dua metode untuk audio yang sama dicatat terpisah di riwayat.

## Cara membaca hasil

Pilih salah satu dari dua metode berikut:

- **Random Forest** menghasilkan nilai `P(berisik)`. Nilai **0,5 atau lebih** diklasifikasikan sebagai **BERISIK**. Ambang RMS tidak memengaruhi hasil metode ini.
- **RMS** membandingkan level rekaman (`dBFS`) dengan ambang. Jika RMS sama dengan atau lebih tinggi daripada ambang, hasilnya **BERISIK**; selain itu **TIDAK BERISIK**. Metode ini tidak menggunakan prediksi Random Forest.

Kartu Random Forest menampilkan probabilitas dan jumlah pohon yang memilih **BERISIK**. Kartu RMS menampilkan level RMS dan ambang yang dibandingkan. Riwayat mencatat metode yang digunakan.

**dBFS bukan pengukuran langsung tingkat kebisingan akustik dalam dB SPL.** Nilainya bergantung pada mikrofon, jarak ke sumber suara, dan pengaturan gain. Ambang awal metode RMS adalah `-30 dBFS`; semakin mendekati nol nilai dBFS, semakin tinggi level digital rekaman. Ambang yang lebih rendah (lebih negatif) membuat lebih banyak rekaman memenuhi aturan RMS, sehingga hasil bisa lebih sering menjadi **BERISIK**.

Model dilatih menggunakan 200 baris data fitur dalam CSV (100 berlabel `Berisik` dan 100 berlabel `Tidak Berisik`). Aplikasi menampilkan akurasi validasi silang 5-fold yang dikelompokkan berdasarkan kolom `grup`. Hasil tetap bersifat indikatif dan dapat keliru, khususnya jika suara atau perangkat rekam berbeda dari data latih.
