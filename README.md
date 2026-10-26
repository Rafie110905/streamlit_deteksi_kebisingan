# Deteksi Kebisingan Ruangan

Aplikasi Streamlit untuk menganalisis suara dan memperkirakan apakah suatu ruangan **berisik** atau **tidak berisik**. Audio dapat diunggah sebagai berkas, direkam melalui mikrofon browser, atau dianalisis secara langsung (realtime) selama mikrofon aktif.

Aplikasi daring: https://psd-deteksi-kebisingan.streamlit.app

## Fitur

- Menerima audio `.mp3`, `.wav`, `.m4a`, `.flac`, dan `.ogg`.
- Tiga sumber audio: unggah berkas, rekam langsung, dan deteksi realtime.
- Tiga pilihan metode: **Random Forest**, **RMS (aturan)**, atau **bandingkan keduanya** pada audio yang sama.
- Menghitung fitur audio: RMS, zero-crossing rate (ZCR), spectral centroid, rolloff, bandwidth, dan flatness.
- Random Forest berisi 300 pohon, dengan keluaran probabilitas dan jumlah pohon yang memilih berisik.
- Metode RMS (aturan) dengan ambang yang dapat diatur.
- Grafik gelombang, spektrogram, serta perbandingan nilai audio terhadap ambang.
- Peringatan suara "Mohon jangan berisik" yang diputar otomatis saat terdeteksi berisik (dapat dimatikan).
- Riwayat pengujian selama sesi aplikasi masih berjalan.

## Struktur berkas

```text
.
├── .streamlit/
│   └── config.toml
├── app_deteksi_kebisingan.py
├── fitur_audio_kebisingan.csv
├── packages.txt
├── peringatan_jangan_berisik.wav
├── requirements.txt
└── README.md
```

`peringatan_jangan_berisik.wav` adalah suara peringatan yang diputar saat hasil berisik. Berkas ini harus berada di folder yang sama dengan `app_deteksi_kebisingan.py`.

`fitur_audio_kebisingan.csv` berisi fitur dan label data latih. Model dilatih ulang setiap kali aplikasi dimulai dari data fitur tersebut. Audio yang diunggah atau direkam tidak disimpan dan tidak dipakai untuk melatih model.

## Persyaratan

- Python dengan paket pada `requirements.txt`.
- Untuk rekaman dan deteksi realtime, browser harus memiliki akses ke mikrofon. Izinkan akses mikrofon untuk situs aplikasi bila diminta.
- Deteksi realtime memakai WebRTC melalui paket `streamlit-webrtc`. Di luar `localhost`, halaman harus dibuka melalui HTTPS.
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

1. Buka tab **Uji sampel**.
2. Pilih **Metode deteksi**: Random Forest, RMS (aturan), atau Bandingkan keduanya.
3. Atur pengaturan metode pada bagian **Pengaturan** bila diperlukan.
4. Pilih **Sumber audio**:
   - **Unggah berkas**: pilih berkas audio.
   - **Rekam langsung**: izinkan mikrofon, lalu rekam suara.
   - **Deteksi realtime**: tekan **START** dan izinkan mikrofon.
5. Untuk unggah dan rekam langsung, analisis berjalan otomatis setelah audio tersedia, tanpa tombol tambahan. Mengubah ambang akan memperbarui hasil dari audio yang sama tanpa memprosesnya ulang.
6. Hasil sebelumnya tercatat di tab **Riwayat**.

### Deteksi realtime

Prediksi pertama muncul setelah 2 detik audio terkumpul, lalu diperbarui setiap detik dari jendela 2 detik terakhir. Tekan **STOP** untuk mengakhiri. Mode ini hanya menampilkan kartu hasil, tanpa grafik.

Bila koneksi tidak terbentuk (misalnya muncul pesan bahwa belum ada audio masuk), penyebab umumnya adalah jaringan yang memblokir koneksi langsung, seperti jaringan kampus, kantor, atau data seluler. Coba jaringan lain, atau tambahkan server TURN pada pengaturan **Secrets** aplikasi:

```toml
[turn]
url = "turn:alamat-server:3478"
username = "nama-pengguna"
credential = "kata-sandi"
```

## Peringatan suara

Kotak centang **Bunyikan peringatan suara bila terdeteksi berisik** (aktif secara bawaan) memutar suara "Mohon jangan berisik" saat hasilnya berisik.

- **Unggah berkas dan rekam langsung:** suara diputar sekali setiap audio baru dianalisis dan hasilnya berisik.
- **Deteksi realtime:** suara diputar saat hasil berisik, lalu diulang bila masih berisik setelah jeda yang diatur (5 sampai 60 detik, bawaan 10 detik).
- **Metode yang menentukan:** pada Random Forest atau RMS, suara mengikuti metode yang dipilih. Pada Bandingkan keduanya, suara dibunyikan bila kedua metode menyatakan berisik.
- Browser dapat menahan pemutaran otomatis sebelum pengguna berinteraksi dengan halaman. Bila suara tidak keluar, tekan tombol putar pada pemutar kecil yang muncul. Pada iPhone/Safari, pemutaran otomatis lebih ketat.
- Pada mode realtime, suara peringatan dapat tertangkap mikrofon dan ikut dihitung sebagai kebisingan. Gunakan headphone atau kecilkan volume.

## Cara membaca hasil

### Random Forest

Model menghasilkan nilai `P(berisik)`. Nilai **0,5 atau lebih** diklasifikasikan sebagai berisik. Kartu hasil menampilkan probabilitas, jumlah pohon yang memilih berisik, dan tingkat keyakinan model.

Pada pengaturan Random Forest tersedia opsi **Perkuat dengan ambang level suara (dBFS)**, aktif secara bawaan:

- Jika model menjawab berisik, hasilnya tetap **berisik**.
- Jika model menjawab tidak berisik, tetapi level rekaman sama dengan atau lebih tinggi dari ambang, hasil akhir menjadi **berisik** dan kartu menampilkan keterangannya.
- Jika tidak memenuhi kedua kondisi tersebut, hasilnya **tidak berisik**.

Ambang awalnya `-30 dBFS`. Semakin mendekati nol, semakin tinggi level digital rekaman. Ambang yang lebih rendah (lebih negatif) membuat hasil lebih sering menjadi berisik. Matikan opsi ini untuk melihat keputusan Random Forest murni.

### RMS (aturan)

Metode ini tidak memerlukan pelatihan. Setiap frame audio dinilai bising bila memenuhi salah satu kondisi berikut:

- RMS mencapai ambang RMS, atau
- RMS mencapai 0,7 kali ambang dan spectral centroid mencapai ambang centroid, atau
- RMS mencapai 0,5 kali ambang dan ZCR mencapai ambang ZCR.

Hasil per frame dihaluskan dengan jendela 5 frame. Audio dinyatakan **berisik** bila skor berisik mencapai ambang skor, atau persentase durasi bising mencapai ambang persen.

| Pengaturan | Nilai awal |
| --- | --- |
| Ambang RMS | 0,050 |
| Ambang centroid | 1600 Hz |
| Ambang ZCR | 0,040 |
| Ambang skor berisik | 0,70 |
| Ambang durasi bising | 40% |

Skor berisik dihitung dari rata-rata RMS, centroid, dan ZCR: `(RMS / 0,08) x 0,45 + (centroid / 2200) x 0,35 + (ZCR / 0,05) x 0,20`.

### Bandingkan keduanya

Kedua kartu hasil ditampilkan berdampingan, disertai keterangan apakah kedua metode sepakat. Tab riwayat mencatat vonis dan skor kedua metode.

## Catatan akurasi

- Random Forest dilatih dari 200 baris data fitur (100 berlabel `Berisik` dan 100 `Tidak Berisik`). Akurasi validasi silang 5-fold yang dikelompokkan berdasarkan kolom `grup` ditampilkan di bilah samping aplikasi (sekitar 76,5%).
- Metode RMS (aturan) dengan ambang awal menghasilkan akurasi sekitar 69,5% pada 200 berkas yang sama.
- Model dilatih dari rekaman berdurasi sekitar 30 detik, sedangkan deteksi realtime hanya melihat 2 detik terakhir, sehingga akurasi realtime cenderung lebih rendah daripada analisis berkas utuh.
- **dBFS bukan pengukuran langsung tingkat kebisingan akustik dalam dB SPL.** Nilainya bergantung pada mikrofon, jarak ke sumber suara, dan pengaturan gain.
- Hasil bersifat indikatif dan dapat keliru, khususnya bila suara atau perangkat rekam berbeda dari data latih.
