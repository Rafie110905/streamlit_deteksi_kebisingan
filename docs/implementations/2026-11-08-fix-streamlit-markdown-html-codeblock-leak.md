# Dokumentasi Perbaikan: Eliminasi Kebocoran Code Block HTML pada Streamlit Native & Admin Panel

**Tanggal**: 8 November 2026  
**Status**: Selesai & Terverifikasi (100% Lulus Uji)  
**Komponen**: Streamlit Frontend (`app_deteksi_kebisingan.py`), Docker Runtime  

---

## 1. Ringkasan Masalah
Pada antarmuka Streamlit (baik kartu status deteksi utama maupun sub-tab Panel Admin seperti *Live Monitoring* dan *Audit Log & Ekspor*), sejumlah elemen grafis HTML bocor dan tampil sebagai blok kode mentah berlatar belakang hitam (`<pre><code>`), bukan sebagai elemen web ter-render.

Elemen yang terdampak antara lain:
1. **Digital VU Meter (Level Suara)** pada kartu hero status admin, menampilkan:
   ```html
   <!-- Digital VU Meter -->
   <div style="margin-top:16px;max-width:560px">
       <div style="display:flex;justify-content:space-between;...
   ```
2. **Tabel Audit Log & Ekspor** dan **Tabel Karakteristik Sumber**, menampilkan baris tabel mentah:
   ```html
   <tr>
       <td style="padding:10px 12px;border-bottom:1px solid #24324f;...
   ```
3. **Kartu Vonis Model Klasifikasi** (`kartu_hasil_model` dan `kartu_hasil_rms`).

---

## 2. Analisis Akar Masalah (Root Cause)
Streamlit merender teks menggunakan spesifikasi **CommonMark Markdown** melalui pemanggilan `st.markdown(html, unsafe_allow_html=True)`.

Menurut standar CommonMark:
- Baris apa pun yang memiliki **4 spasi indentasi atau lebih** setelah baris kosong dianggap sebagai **Indented Code Block**.
- Di Python, string multiline (`f"""..."""`) yang ditulis di dalam fungsi atau blok indentasi (`with`, `def`, `if`) mewarisi spasi indentasi Python (biasanya 16 hingga 28 spasi pada awal baris).
- Akibatnya, CommonMark secara otomatis membungkus baris-baris berindentasi tersebut ke dalam tag `<pre><code>...</code></pre>`, sehingga browser menampilkannya sebagai teks kode hitam mentah alih-alih merender elemen DOM visual HTML.

---

## 3. Solusi yang Diterapkan

Dibuat fungsi utilitas terpusat `_st_html` pada tingkat global `app_deteksi_kebisingan.py`:

```python
def _st_html(html_str: str):
    """Render HTML di Streamlit secara aman tanpa risiko terdeteksi sebagai Markdown code block."""
    text = re.sub(r"<!--.*?-->", "", str(html_str), flags=re.DOTALL)
    clean = " ".join(line.strip() for line in text.splitlines() if line.strip())
    st.markdown(clean, unsafe_allow_html=True)
```

### Mekanisme Kerja:
1. **Pembersihan Komentar**: Menghapus seluruh komentar HTML seperti `<!-- Digital VU Meter -->` menggunakan regex `re.DOTALL`.
2. **Penghapusan Indentasi (*Leading Whitespace Removal*)**: Memotong semua spasi dan tab di awal dan akhir setiap baris (`line.strip()`).
3. **Penyatuan Segmen Baris Tunggal (*Token Flattening*)**: Menggabungkan seluruh baris menjadi satu aliran token HTML tunggal dengan pemisah spasi (`" ".join(...)`), menyaring baris-baris kosong.
4. **Jaminan Anti-Codeblock**: Karena tidak ada satu pun baris yang diawali dengan 4 spasi, parser CommonMark tidak akan pernah memicu aturan indented code block.
5. **Migrasi Seluruh Komponen**: Seluruh pemanggilan `st.markdown(..., unsafe_allow_html=True)` pada:
   - `kartu_hasil_model` (deteksi realtime & offline)
   - `kartu_hasil_rms` (deteksi realtime & offline)
   - `_hero_status_html`
   - `_kartu_kpi_html`
   - Tabel audit log, perbandingan sumber, dan sesi klien
   - Kartu arsitektur model dan status server/database
   dialihkan 100% menggunakan `_st_html(...)`.

---

## 4. Berkas yang Diubah
- [app_deteksi_kebisingan.py](file:///c:/Users/dedyn/Downloads/psd/psd%20tugas/streamlit_deteksi_kebisingan/app_deteksi_kebisingan.py):
  - Penambahan pustaka `re`.
  - Definisi global `_st_html`.
  - Migrasi seluruh blok kartu dan tabel HTML ke `_st_html`.
- Kontainer Docker `streamlit_deteksi_kebisingan-deteksi-kebisingan-1`:
  - Berkas disinkronkan dan kontainer di-*restart*.

---

## 5. Hasil Pengujian & Verifikasi Mandiri
1. **Syntax & Compilation Check**:
   - `python -m py_compile app_deteksi_kebisingan.py` lolos tanpa galat (exit code 0).
2. **Container Logs Verification**:
   - Kontainer Streamlit berhasil me-*reload* aplikasi dan berjalan normal pada port `8011` / `8501`.
3. **HTTP Health Check**:
   - `curl.exe -I http://localhost:8011` mengembalikan status `HTTP/1.1 200 OK`.
4. **Sanitizer Unit Test**:
   - Skrip pengujian `test_sanitizer.py` memvalidasi bahwa blok multiline berindentasi 24 spasi dan komentar HTML tereduksi sempurna menjadi token datar tanpa indentasi awal.
5. **Zero-Trust Security Pentest**:
   - Endpoint admin `/api/status` dan `/embed` tetap menolak akses tanpa token (401 Unauthorized), menjamin tidak ada regresi keamanan.
