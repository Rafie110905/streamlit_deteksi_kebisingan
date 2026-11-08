# Catatan Temuan & Analisis Anomali Data Fitur Audio

Dokumen ini mencatat hasil audit kualitas data (*data quality audit*) pada dataset fitur audio kebisingan (`fitur_audio_kebisingan.csv`).

---

## 1. Ringkasan Dataset
* **Total Baris Awal**: 200 baris
* **Jumlah Fitur**: 24 fitur audio (ekstraksi Librosa: RMS, ZCR, Spectral Centroid, Rolloff, Bandwidth, Flatness) + metadata (`file_name`, `label`, `grup`, `rms_dbfs`)
* **Distribusi Kelas Awal**:
  * `Berisik`: 100 sampel (50%)
  * `Tidak Berisik`: 100 sampel (50%)
* **Jumlah Grup Rekaman**: 35 kelompok perekam

---

## 2. Temuan Anomali

### A. Kontradiksi Label Fatal (Vektor Fitur Identik, Label Berlawanan)
Ditemukan 2 file yang memiliki nilai seluruh 24 fitur audio numerik yang **100% identik**, namun diberi label berlawanan:
* **Indeks 18**: `24-034 berisik 1.mp3` | Grup: `24-034` | Label: **Berisik** (RMS: `0.011537`, dBFS: `-37.77 dB`)
* **Indeks 117**: `24-034 non-berisik 1.mp3` | Grup: `24-034` | Label: **Tidak Berisik** (RMS: `0.011537`, dBFS: `-37.77 dB`)

> **Dampak ke Model**: Model klasifikasi tidak dapat membedakan kedua sampel ini. Salah satu atau keduanya dipastikan akan selalu salah diprediksi. Menghapus kedua baris ambigu ini adalah tindakan terbaik untuk menjaga integritas data latih.

---

### B. Duplikasi Fitur / Kloning Data (23 Baris Duplikat)
Terdapat 23 baris yang nilainya sama persis dengan baris lain:
1. **Duplikasi Dalam Satu Grup (File yang sama dinamai ulang)**:
   * Grup `24-157`: `24-157 Berisik 1.mp3`, `24-157 Berisik 2.mp3`, dan `24-157 Berisik 3.mp3` memiliki nilai fitur identik.
   * Grup `24-078`: `NonBerisik 1`, `NonBerisik 2`, dan `NonBerisik 3` identik.
   * Grup `24-196`: `non berisik 1`, `non berisik 2`, dan `non berisik 3` identik.
2. **Duplikasi Lintas Grup (Data grup lain disalin)**:
   * Grup `24-071` dan `24-098`: Ketiga sampel berisik di grup `24-071` identik dengan ketiga sampel di `24-098`.
   * Sampel `24-003 berisik` identik dengan `24-053 berisik 3` dan `24-079 berisik (1, 2)`.

> **Dampak ke Model**: Menyebabkan *data leakage* dan overfitting semu (model seolah-olah memiliki akurasi tinggi karena menguji data yang identik dengan data latih).

---

### C. Dugaan Label Tertukar (*Inverted Labels*)
* **Kasus Ekstrem Grup `24-168`**:
  * `24-168_Berisik.mp3` memiliki RMS sangat rendah: **-39.37 dBFS** (sangat sunyi).
  * `24-168_Non-Berisik.mp3` memiliki RMS sangat tinggi: **-16.92 dBFS** (sangat bising).
  * Pada evaluasi Cross-Validation, ketiga model (KNN, SVM, Random Forest) salah 100% dalam memprediksi grup ini karena labelnya terbalik.
* Terdapat **10 sampel** "Tidak Berisik" yang level energinya (dBFS) lebih tinggi dari median sampel "Berisik" global (-21.30 dBFS).

---

### D. Disparitas Sensitivitas / Gain Mikrofon (*Domain Shift*)
* Setiap kelompok menggunakan perangkat mikrofon dan pengaturan *gain* yang berbeda-beda.
* Contoh: Pada grup `24-125`, suara paling tenang (`Non-Berisik-1.mp3`) berada di level **-14.17 dBFS**, jauh lebih keras dari suara "Berisik" milik grup `24-034` (-37.77 dBFS).
* Jika model hanya mengandalkan fitur energi (RMS/dBFS), model rentan keliru saat menghadapi audio dari perangkat baru.

---

### E. Outlier Nilai (*Mathematical / Signal Anomaly*)
* **Flatness Max = 1.000001**:
  File `24-003 berisik (2).mp3`, `(3).mp3`, dan `24-027_berisik1.mp3` memiliki nilai maksimum *spectral flatness* sebesar `1.0`. Menandakan sinyal mengalami kliping digital ekstrem atau *pure noise*.
* **RMS Outliers**:
  9 file memiliki `rms_mean` > `0.20` (outlier berdasarkan batas atas IQR $Q3 + 1.5 \times IQR$).

---

## 3. Struktur File Dataset Hasil Kloning

Untuk menjaga keamanan data dan reprodusibilitas eksperimen, dataset telah dipisahkan menjadi dua berkas:

1. **`fitur_audio_kebisingan_asli.csv`**
   * Salinan lengkap dari data awal (200 baris, 100 Berisik, 100 Tidak Berisik).
   * Digunakan sebagai data acuan/baseline historis.

2. **`fitur_audio_kebisingan_tanpa_kontradiksi.csv`**
   * Data setelah baris kontradiktif (indeks 18 dan 117: `24-034 berisik 1.mp3` dan `24-034 non-berisik 1.mp3`) dihapus.
   * Total baris: 198 baris (99 Berisik, 99 Tidak Berisik).
   * Menghilangkan ambiguitas fitur identik dengan label berlawanan pada data latih.

