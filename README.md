# Deteksi Kebisingan Ruangan

Aplikasi Streamlit untuk menganalisis rekaman audio dan memperkirakan apakah suara ruangan **berisik** atau **tidak berisik**. Audio bisa diunggah atau direkam langsung melalui mikrofon browser.

## Fitur

- Menerima audio `.mp3`, `.wav`, `.m4a`, `.flac`, dan `.ogg`.
- Merekam audio langsung dari mikrofon melalui browser.
- Menghitung fitur audio seperti RMS, zero-crossing rate, spectral centroid, rolloff, bandwidth, dan flatness.
- Mengklasifikasikan audio dengan Random Forest berisi 300 pohon.
- Menampilkan probabilitas model, suara pohon, grafik gelombang, dan spektrogram.
- Menyimpan hasil pengujian selama sesi aplikasi masih berjalan.
- Menyediakan ambang RMS yang dapat diatur untuk menentukan hasil **BERISIK** ketika level rekaman mencapai ambang tersebut.

## Struktur berkas

```text
.
├── app_deteksi_kebisingan.py
├── fitur_audio_kebisingan.csv
├── packages.txt
└── requirements.txt
```

`fitur_audio_kebisingan.csv` berisi fitur dan label data latih yang diperlukan aplikasi. Aplikasi tidak melatih model dari audio yang diunggah; model dilatih saat aplikasi dimulai menggunakan data fitur tersebut.

## Persyaratan

- Python dengan paket pada `requirements.txt`.
- Untuk perekaman langsung, browser harus memiliki akses ke mikrofon. Jika diminta, izinkan akses mikrofon untuk situs aplikasi.
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

## Menguji audio

1. Buka tab **Uji Sampel**.
2. Atur **Ambang suara keras (RMS, dBFS)** bila diperlukan.
3. Pilih **Unggah berkas** dan pilih audio, atau pilih **Rekam langsung**.
4. Untuk rekaman langsung, izinkan akses mikrofon pada browser dan lakukan rekaman.
5. Klik **Jalankan hutan keputusan** untuk melihat hasil dan grafik.

## Cara membaca hasil

Random Forest menghasilkan nilai `P(berisik)`. Nilai **0,5 atau lebih** diklasifikasikan model sebagai berisik. Aplikasi juga menggunakan aturan ambang RMS:

- Jika prediksi model sudah **BERISIK**, hasilnya tetap **BERISIK**.
- Jika prediksi model **TIDAK BERISIK**, tetapi RMS rekaman sama dengan atau lebih tinggi daripada ambang, hasil akhir menjadi **BERISIK**.
- Jika tidak memenuhi kedua kondisi tersebut, hasil akhirnya **TIDAK BERISIK**.

Kartu hasil menampilkan probabilitas dan jumlah pohon yang memilih **BERISIK** dari model Random Forest. Jika ambang RMS yang mengubah hasil akhir, kartu juga menampilkan keterangan tersebut. Riwayat mencatat metode keputusan yang digunakan.

**dBFS bukan pengukuran langsung tingkat kebisingan akustik dalam dB SPL.** Nilainya bergantung pada mikrofon, jarak ke sumber suara, dan pengaturan gain. Ambang awal aplikasi adalah `-30 dBFS`; semakin mendekati nol nilai dBFS, semakin tinggi level digital rekaman. Ambang yang lebih rendah (lebih negatif) membuat lebih banyak rekaman memenuhi aturan RMS, sehingga hasil bisa lebih sering menjadi **BERISIK**.

Model dilatih menggunakan 200 baris data fitur dalam CSV (100 berlabel `Berisik` dan 100 berlabel `Tidak Berisik`). Aplikasi menampilkan akurasi validasi silang 5-fold yang dikelompokkan berdasarkan kolom `grup`. Hasil tetap bersifat indikatif dan dapat keliru, khususnya jika suara atau perangkat rekam berbeda dari data latih.
