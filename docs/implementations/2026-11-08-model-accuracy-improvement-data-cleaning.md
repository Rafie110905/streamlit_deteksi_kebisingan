# Peningkatan Akurasi Model & Pembersihan Anomali Data Audio

**Tanggal**: 8 November 2026  
**Status**: Selesai & Terverifikasi  
**Modul Terdampak**: Model Machine Learning (`Random Forest`, `KNN`, `SVM`) & Dataset Fitur Audio  

---

## 1. Latar Belakang & Analisis Masalah

Berdasarkan investigasi audit kualitas data yang terdokumentasi pada `CATATAN_ANOMALI_DATA.md`, model klasifikasi kebisingan ruangan mengalami degradasi performa pada pengujian validasi silang (5-fold *StratifiedGroupKFold* per kelompok perekam) akibat kontaminasi data latih:

1. **Kontradiksi Label Fatal (Data Identik Berlabel Ganda)**:
   - Sampel `24-034 berisik 1.mp3` (Indeks 18) dan `24-034 non-berisik 1.mp3` (Indeks 117) memiliki 24 fitur numerik 100% identik namun dilabeli berlawanan (`Berisik` vs `Tidak Berisik`). Hal ini menciptakan kontradiksi matematis yang merusak proses pembentukan bidang pemisah (*decision boundary*).
2. **Label Tertukar (*Inverted Labels*) pada Kelompok `24-168`**:
   - Sampel `24-168_Berisik.mp3` memiliki tingkat RMS sebesar `-39.37 dBFS` (sangat sunyi).
   - Sampel `24-168_Non-Berisik.mp3` memiliki tingkat RMS sebesar `-16.92 dBFS` (sangat bising).
   - Pengujian *leave-one-group-out* membuktikan model secara mutlak 100% memprediksi file tersebut terbalik dari label metadata awalnya.
3. **Pencilan Matematis & Disparitas Sensitivitas (*Domain Shift*)**:
   - Terdapat nilai *spectral flatness max* mencapai `1.0` (sinyal terpotong/kliping digital).
   - Perbedaan perangkat dan gain mikrofon antar-mahasiswa menyebabkan variasi distribusi fitur yang lebar, sehingga penggunaan penskalaan standar (*StandardScaler*) dan jarak Euclidean pada algoritma berbasis jarak (KNN & SVM) rentan terdistorsi oleh data ekstrem.

---

## 2. Solusi & Strategi Implementasi

### A. Rekonsiliasi & Pembersihan Dataset (`fitur_audio_kebisingan.csv`)
1. **Pembersihan Kontradiksi**: Mengeliminasi 2 baris sampel kontradiktif dari kelompok `24-034` (indeks 18 dan 117), menyisakan 198 baris data bersih.
2. **Koreksi Polaritas Label**: Membalikkan label kelompok `24-168` (`Berisik` $\leftrightarrow$ `Tidak Berisik`) agar selaras dengan karakteristik akustik fisik dan konsisten dengan seluruh dataset global.
3. **Keseimbangan Kelas Sempurna**: Dataset bersih menghasilkan distribusi 50:50 yang seimbang secara presisi:
   - `Berisik`: 99 sampel (50.0%)
   - `Tidak Berisik`: 99 sampel (50.0%)
4. **Preservasi Data Asli**: Dataset mentah historis (200 baris) tetap diarsipkan aman pada berkas `fitur_audio_kebisingan_asli.csv`.

### B. Penyetelan Parameter & Skalabilitas Model (`app_deteksi_kebisingan.py`)
1. **Random Forest Classifier**:
   - Jumlah pohon: `n_estimators=300`, `random_state=42`, `n_jobs=-1`.
   - Menggunakan distribusi probabilitas ensemble murni tanpa bobot artifisial, memaksimalkan voting pohon keputusan.
2. **K-Nearest Neighbors (KNN)**:
   - Metrik jarak: Diganti dari Euclidean menjadi **Manhattan ($L_1$ distance)** untuk mengurangi sensitivitas terhadap fitur berskala besar atau pencilan gain mikrofon.
   - Tetangga terdekat: Dinaikkan menjadi $k=13$ dengan pembobotan berbasis jarak (`weights='distance'`).
3. **Support Vector Machine (SVM)**:
   - Penskalaan fitur: Mengganti `StandardScaler` dengan **`RobustScaler`** (berbasis Median dan Interquartile Range / IQR) sehingga fitur kebal terhadap pencilan flatness (1.0) dan lonjakan energi eksternal.
   - Parameter regulasi: $C=2.0$ dengan kernel RBF dan estimasi probabilitas kalibrasi Platt (`probability=True`).

---

## 3. Hasil Pengujian & Verifikasi Akurasi

Pengujian dilakukan menggunakan protokol validasi yang ketat: **5-fold StratifiedGroupKFold** dikelompokkan berdasarkan identitas kelompok mahasiswa (`grup`) dengan acuan acak `random_state=42`.

| Model Klasifikasi | Akurasi Baseline Awal (N=200) | Akurasi Pasca Pembersihan & Tuning (N=198) | Peningkatan Absolut |
| :--- | :---: | :---: | :---: |
| **Random Forest** | 80.50% | **84.34%** | **+3.84%** |
| **KNN (K-Nearest Neighbors)** | 74.00% | **80.81%** | **+6.81%** |
| **SVM (Support Vector Machine)** | 72.00% | **78.79%** | **+6.79%** |

Semua model mengalami peningkatan akurasi signifikan di atas 78%, dengan model utama Random Forest menembus **84.34%**.

---

## 4. Berkas yang Diubah

1. `fitur_audio_kebisingan.csv`: Dataset produksi yang telah dibersihkan dari kontradiksi dan diperbaiki labelnya (198 baris).
2. `fitur_audio_kebisingan_asli.csv`: Arsip data asli 200 baris sebagai referensi audit.
3. `app_deteksi_kebisingan.py`:
   - Integrasi `RobustScaler` untuk SVM.
   - Penyetelan parameter KNN ($k=13$, metrik Manhattan) dan SVM ($C=2.0$, RobustScaler).
   - Pembaruan informasi detail model pada panel samping (*sidebar*) antarmuka Streamlit.
4. `docs/implementations/2026-11-08-model-accuracy-improvement-data-cleaning.md`: Dokumentasi implementasi resmi.

