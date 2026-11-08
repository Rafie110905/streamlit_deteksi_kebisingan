# Panduan Build Docker & Konfigurasi Port

Dokumen ini menjelaskan cara membangun (*build*) dan menjalankan seluruh layanan aplikasi menggunakan Docker, beserta penjelasan lengkap konfigurasi port yang digunakan.

---

## Arsitektur Layanan

Aplikasi terdiri dari **dua service** yang berjalan bersamaan melalui Docker Compose:

```
┌─────────────────────────────────────────────────────────────┐
│                      Host Machine                           │
│                                                             │
│   Browser → :8011 ──────► Streamlit (container :8501)      │
│   Browser → :8000 ──────► Admin Panel FastAPI (:8000)      │
│                                │                            │
│                        [Volume: noise_data]                 │
│                        deteksi_log.sqlite                   │
└─────────────────────────────────────────────────────────────┘
```

| Service | Image | Port Host | Port Container | Fungsi |
|---------|-------|-----------|----------------|--------|
| `deteksi-kebisingan` | `deteksi-kebisingan:latest` | **8011** | 8501 | Aplikasi Streamlit utama |
| `admin-panel` | `admin-kebisingan:latest` | **8000** | 8000 | Dashboard admin (FastAPI) |

---

## Prasyarat

- **Docker Desktop** versi 24+ terpasang dan berjalan
- **Docker Compose** v2 (sudah termasuk dalam Docker Desktop)
- Port `8011` dan `8000` pada host tidak digunakan oleh aplikasi lain

---

## Struktur File Docker

```
streamlit_deteksi_kebisingan/
├── Dockerfile                  ← Dockerfile untuk Streamlit
├── docker-compose.yml          ← Orkestrasi kedua service
├── admin_panel/
│   ├── Dockerfile              ← Dockerfile untuk Admin Panel
│   └── requirements.txt        ← Dependensi FastAPI + Uvicorn
├── requirements.txt            ← Dependensi Streamlit
└── packages.txt                ← Paket sistem Linux (ffmpeg, libsndfile1)
```

---

## Cara Menjalankan

### 1. Build + Jalankan Semua Service (Perintah Utama)

```powershell
docker compose up --build
```

Perintah ini akan:
1. Membangun image `deteksi-kebisingan:latest` dari `Dockerfile`
2. Membangun image `admin-kebisingan:latest` dari `admin_panel/Dockerfile`
3. Membuat shared volume `noise_data` untuk database SQLite
4. Menjalankan kedua container

**Akses setelah jalan:**
- 🎙️ Streamlit → [http://localhost:8011](http://localhost:8011)
- 🛠️ Admin Panel → [http://localhost:8000](http://localhost:8000) *(login: `admin` / `admin123`)*

### 2. Jalankan Tanpa Build Ulang (Jika Image Sudah Ada)

```powershell
docker compose up
```

### 3. Jalankan di Background (Detached Mode)

```powershell
docker compose up --build -d
```

Cek status container:
```powershell
docker compose ps
```

### 4. Hentikan Semua Service

```powershell
docker compose down
```

Hentikan dan **hapus volume** (data log deteksi akan terhapus):
```powershell
docker compose down -v
```

---

## Detail Konfigurasi Port

### Streamlit (Service: `deteksi-kebisingan`)

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| Port host | **8011** | Port yang diakses dari browser |
| Port container | **8501** | Port default Streamlit di dalam container |
| Mapping | `8011:8501` | Host:Container |

> [!NOTE]
> Port host **8011** dipilih khusus untuk kelompok 11 (sesuai konvensi proyek). Jika port 8011 sudah dipakai, ubah angka pertama di `docker-compose.yml` pada bagian `ports`.

Konfigurasi server Streamlit ada di CMD pada `Dockerfile`:
```dockerfile
CMD ["python", "-m", "streamlit", "run", "app_deteksi_kebisingan.py",
     "--server.address=0.0.0.0",
     "--server.port=8501",
     "--server.headless=true"]
```

**Penjelasan flag:**
- `--server.address=0.0.0.0` → Streamlit mendengarkan dari semua interface (wajib untuk Docker)
- `--server.port=8501` → Port internal container
- `--server.headless=true` → Nonaktifkan pembukaan browser otomatis

### Admin Panel (Service: `admin-panel`)

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| Port host | **8000** | Port yang diakses dari browser |
| Port container | **8000** | Port Uvicorn di dalam container |
| Mapping | `8000:8000` | Host:Container |

Konfigurasi CMD pada `admin_panel/Dockerfile`:
```dockerfile
CMD ["uvicorn", "admin_panel.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Mengubah Port Host

Jika port `8011` atau `8000` sudah digunakan, edit bagian `ports` di [`docker-compose.yml`](docker-compose.yml):

```yaml
services:
  deteksi-kebisingan:
    ports:
      - "PORT_BARU:8501"   # ← ganti PORT_BARU sesuai kebutuhan

  admin-panel:
    ports:
      - "PORT_BARU:8000"   # ← ganti PORT_BARU sesuai kebutuhan
```

> [!WARNING]
> Jangan ubah angka **setelah titik dua** (port container). Hanya angka **sebelum titik dua** (port host) yang boleh diganti.

---

## Konfigurasi Tema Streamlit

Tema gelap dikonfigurasi di [`.streamlit/config.toml`](.streamlit/config.toml):

```toml
[theme]
base                  = "dark"
primaryColor          = "#ef4444"   # Merah — warna aksen utama
backgroundColor       = "#090909"   # Hitam gelap — latar halaman
secondaryBackgroundColor = "#171717" # Abu gelap — latar panel/kartu
textColor             = "#f5f5f5"   # Putih — warna teks
font                  = "sans serif"
```

---

## Variabel Environment

Kredensial admin panel dikonfigurasi via environment variable di `docker-compose.yml`. Nilai default bisa diubah dengan membuat file [`.env`](.env) di root proyek:

```env
ADMIN_USER=admin
ADMIN_PASS=admin123
```

> [!CAUTION]
> File `.env` sudah masuk daftar `.gitignore` dan **tidak akan ikut ke-push ke GitHub**. Jangan pernah hardcode password di kode sumber.

---

## Shared Volume (Database SQLite)

Kedua service berbagi satu volume Docker bernama `noise_data`:

```yaml
volumes:
  noise_data:
```

- **Streamlit** → menulis hasil deteksi ke `deteksi_log.sqlite` di dalam volume
- **Admin Panel** → membaca dari `deteksi_log.sqlite` yang sama

Data log tetap tersimpan meskipun container di-restart, selama volume tidak dihapus (`docker compose down` tanpa flag `-v`).

---

## Troubleshooting

### Port sudah digunakan
```
Error: Bind for 0.0.0.0:8011 failed: port is already allocated
```
→ Ganti port host di `docker-compose.yml` atau hentikan proses yang memakai port tersebut.

### Izin mikrofon untuk WebRTC Realtime
WebRTC membutuhkan koneksi **HTTPS** jika diakses bukan dari `localhost`. Jika deploy ke server publik, pastikan menggunakan HTTPS (misalnya via Nginx reverse proxy + SSL/Let's Encrypt).

### Admin Panel tidak bisa baca data Streamlit
Pastikan kedua service berjalan bersamaan dengan `docker compose up` (bukan masing-masing `docker run`). Volume `noise_data` hanya terbuat otomatis saat menggunakan Compose.

### Build ulang setelah ada perubahan kode
```powershell
docker compose up --build
```
Flag `--build` memaksa Docker membangun ulang image meskipun sudah ada image lama.
