"""
deteksi_logger.py
Modul logging terpusat — menulis setiap hasil deteksi ke SQLite.
Digunakan oleh app_deteksi_kebisingan.py dan dibaca oleh admin_panel.
"""

import os
import sqlite3
import threading
from datetime import datetime

# Jika dijalankan via Docker, DB disimpan di volume bersama (DB_DIR env)
_db_dir = os.getenv("DB_DIR", os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_db_dir, "deteksi_log.sqlite")
_lock = threading.Lock()


def _koneksi():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def inisialisasi_db():
    """Buat tabel jika belum ada. Dipanggil sekali saat startup."""
    with _lock:
        conn = _koneksi()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS deteksi (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                sumber      TEXT    NOT NULL,
                metode      TEXT    NOT NULL,
                vonis       TEXT    NOT NULL,
                p_berisik   REAL,
                rms_dbfs    REAL,
                nama_file   TEXT,
                keyakinan   TEXT,
                sesi_id     TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sesi_aktif (
                sesi_id     TEXT PRIMARY KEY,
                mulai       TEXT NOT NULL,
                terakhir    TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()


def log_deteksi(
    sumber: str,       # "Upload", "Rekam Langsung", "Realtime WebRTC"
    metode: str,       # "Random Forest", "KNN", "SVM", "RMS"
    vonis: str,        # "BERISIK" / "TIDAK BERISIK"
    p_berisik=None,    # float 0-1 (hanya model)
    rms_dbfs=None,     # float dBFS
    nama_file=None,    # str nama file
    keyakinan=None,    # str tingkat keyakinan
    sesi_id=None,      # str session_id Streamlit
):
    """Tulis satu baris hasil deteksi ke SQLite."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _lock:
        try:
            conn = _koneksi()
            conn.execute("""
                INSERT INTO deteksi
                    (timestamp, sumber, metode, vonis, p_berisik, rms_dbfs, nama_file, keyakinan, sesi_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (ts, sumber, metode, vonis, p_berisik, rms_dbfs, nama_file, keyakinan, sesi_id))
            conn.commit()
            conn.close()
        except Exception:
            pass  # Jangan crash aplikasi utama jika logging gagal


def daftar_sesi(sesi_id: str):
    """Catat sesi Streamlit aktif (dipanggil setiap run)."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _lock:
        try:
            conn = _koneksi()
            conn.execute("""
                INSERT INTO sesi_aktif (sesi_id, mulai, terakhir)
                VALUES (?, ?, ?)
                ON CONFLICT(sesi_id) DO UPDATE SET terakhir=excluded.terakhir
            """, (sesi_id, ts, ts))
            conn.commit()
            conn.close()
        except Exception:
            pass


def hapus_sesi_mati(batas_detik: int = 60):
    """Hapus sesi yang tidak aktif lebih dari batas_detik detik."""
    from datetime import timedelta
    batas = (datetime.now() - timedelta(seconds=batas_detik)).strftime("%Y-%m-%d %H:%M:%S")
    with _lock:
        try:
            conn = _koneksi()
            conn.execute("DELETE FROM sesi_aktif WHERE terakhir < ?", (batas,))
            conn.commit()
            conn.close()
        except Exception:
            pass


def ambil_riwayat(limit: int = 100):
    """Ambil riwayat deteksi terbaru untuk admin panel."""
    with _lock:
        try:
            conn = _koneksi()
            rows = conn.execute(
                "SELECT * FROM deteksi ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception:
            return []


def ambil_status_terkini():
    """Ambil 1 baris deteksi terbaru beserta metadata keaktifan streaming realtime."""
    with _lock:
        try:
            conn = _koneksi()
            row = conn.execute(
                "SELECT * FROM deteksi ORDER BY id DESC LIMIT 1"
            ).fetchone()
            conn.close()
            if not row:
                return None
            data = dict(row)

            # Hitung selisih detik dari waktu saat ini
            try:
                t_row = datetime.strptime(data["timestamp"], "%Y-%m-%d %H:%M:%S")
                detik_lalu = int((datetime.now() - t_row).total_seconds())
                if detik_lalu < 0:
                    detik_lalu = 0
            except Exception:
                detik_lalu = 999999

            data["detik_lalu"] = detik_lalu

            if detik_lalu <= 3:
                data["keterangan_waktu"] = "Baru saja"
            elif detik_lalu < 60:
                data["keterangan_waktu"] = f"{detik_lalu} detik lalu"
            elif detik_lalu < 3600:
                data["keterangan_waktu"] = f"{detik_lalu // 60} menit lalu"
            elif detik_lalu < 86400:
                data["keterangan_waktu"] = f"{detik_lalu // 3600} jam lalu"
            else:
                data["keterangan_waktu"] = f"{detik_lalu // 86400} hari lalu"

            is_realtime_sumber = "realtime" in str(data.get("sumber", "")).lower()
            data["is_realtime_sumber"] = is_realtime_sumber

            # Ambang batas toleransi: 6 detik untuk streaming WebRTC aktif
            if is_realtime_sumber and detik_lalu <= 6:
                data["is_live"] = True
                data["status_aliran"] = "LIVE_AKTIF"
                data["status_teks"] = "REALTIME AKTIF (ON)"
            elif is_realtime_sumber:
                data["is_live"] = False
                data["status_aliran"] = "REALTIME_TERHENTI"
                data["status_teks"] = "REALTIME NONAKTIF (DATA LAMA)"
            else:
                data["is_live"] = False
                data["status_aliran"] = "NON_REALTIME"
                data["status_teks"] = f"UJI SAMPEL ({data.get('sumber', 'MANUAL')})"

            return data
        except Exception:
            return None


def ambil_data_grafik(menit: int = 10):
    """Ambil data deteksi N menit terakhir untuk grafik level suara."""
    from datetime import timedelta
    batas = (datetime.now() - timedelta(minutes=menit)).strftime("%Y-%m-%d %H:%M:%S")
    with _lock:
        try:
            conn = _koneksi()
            rows = conn.execute(
                "SELECT timestamp, rms_dbfs, vonis FROM deteksi WHERE timestamp >= ? ORDER BY id ASC",
                (batas,)
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception:
            return []


def hitung_statistik():
    """Hitung statistik ringkasan untuk dashboard admin."""
    with _lock:
        try:
            conn = _koneksi()
            total = conn.execute("SELECT COUNT(*) FROM deteksi").fetchone()[0]
            berisik = conn.execute(
                "SELECT COUNT(*) FROM deteksi WHERE vonis='BERISIK'"
            ).fetchone()[0]
            hari_ini = conn.execute(
                "SELECT COUNT(*) FROM deteksi WHERE DATE(timestamp)=DATE('now','localtime')"
            ).fetchone()[0]
            berisik_hari_ini = conn.execute(
                "SELECT COUNT(*) FROM deteksi WHERE vonis='BERISIK' AND DATE(timestamp)=DATE('now','localtime')"
            ).fetchone()[0]
            sesi = conn.execute("SELECT COUNT(*) FROM sesi_aktif").fetchone()[0]

            # Rincian per sumber deteksi
            total_realtime = conn.execute(
                "SELECT COUNT(*) FROM deteksi WHERE sumber LIKE '%Realtime%'"
            ).fetchone()[0]
            total_rekam = conn.execute(
                "SELECT COUNT(*) FROM deteksi WHERE sumber LIKE '%Rekam%'"
            ).fetchone()[0]
            total_upload = conn.execute(
                "SELECT COUNT(*) FROM deteksi WHERE sumber LIKE '%Upload%' OR sumber LIKE '%Unggah%'"
            ).fetchone()[0]

            from datetime import timedelta
            limit_stream = (datetime.now() - timedelta(seconds=6)).strftime("%Y-%m-%d %H:%M:%S")
            total_streamer_aktif = conn.execute(
                "SELECT COUNT(DISTINCT sesi_id) FROM deteksi WHERE sumber LIKE '%Realtime%' AND timestamp >= ?",
                (limit_stream,)
            ).fetchone()[0]

            conn.close()
            return {
                "total": total,
                "berisik": berisik,
                "tenang": total - berisik,
                "hari_ini": hari_ini,
                "berisik_hari_ini": berisik_hari_ini,
                "sesi_aktif": sesi,
                "persen_berisik": round(berisik / total * 100, 1) if total else 0,
                "total_realtime": total_realtime,
                "total_rekam": total_rekam,
                "total_upload": total_upload,
                "realtime_streaming_aktif": total_streamer_aktif > 0,
                "total_streamer_aktif": total_streamer_aktif,
                "sumber": {
                    "realtime": total_realtime,
                    "rekam": total_rekam,
                    "upload": total_upload,
                },
            }
        except Exception:
            return {}


def ambil_riwayat_terfilter(
    limit: int = 50,
    vonis: str = None,
    metode: str = None,
    query: str = None,
    sumber: str = None,
):
    """Ambil riwayat deteksi dengan filter vonis, metode, sumber, dan pencarian."""
    with _lock:
        try:
            conn = _koneksi()
            sql = "SELECT * FROM deteksi WHERE 1=1"
            params = []
            if vonis and vonis.upper() not in ("SEMUA", "ALL", ""):
                sql += " AND vonis = ?"
                params.append(vonis.upper())
            if metode and metode not in ("Semua", "All", ""):
                sql += " AND metode = ?"
                params.append(metode)
            if sumber and sumber not in ("Semua", "All", ""):
                s_lower = sumber.lower()
                if "realtime" in s_lower:
                    sql += " AND sumber LIKE '%Realtime%'"
                elif "rekam" in s_lower:
                    sql += " AND sumber LIKE '%Rekam%'"
                elif "upload" in s_lower or "unggah" in s_lower:
                    sql += " AND (sumber LIKE '%Upload%' OR sumber LIKE '%Unggah%')"
                else:
                    sql += " AND sumber = ?"
                    params.append(sumber)
            if query and query.strip():
                sql += " AND (nama_file LIKE ? OR sesi_id LIKE ? OR sumber LIKE ?)"
                q = f"%{query.strip()}%"
                params.extend([q, q, q])
            sql += " ORDER BY id DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception:
            return []


def ambil_analisis_mendalam():
    """Ambil data agregat mendalam untuk menu Analisis & Statistik."""
    with _lock:
        try:
            conn = _koneksi()
            # 1. Distribusi per Metode
            metode_rows = conn.execute("""
                SELECT metode,
                       COUNT(*) as total,
                       SUM(CASE WHEN vonis='BERISIK' THEN 1 ELSE 0 END) as berisik,
                       SUM(CASE WHEN vonis='TIDAK BERISIK' THEN 1 ELSE 0 END) as tenang,
                       ROUND(AVG(rms_dbfs), 1) as avg_rms
                FROM deteksi
                GROUP BY metode
            """).fetchall()

            # 2. Distribusi per Sumber (distandarisasi ke 3 kategori utama)
            sumber_rows = conn.execute("""
                SELECT 
                    CASE 
                        WHEN sumber LIKE '%Realtime%' THEN 'Realtime WebRTC'
                        WHEN sumber LIKE '%Rekam%' THEN 'Rekam Langsung'
                        WHEN sumber LIKE '%Upload%' OR sumber LIKE '%Unggah%' THEN 'Upload Berkas'
                        ELSE sumber 
                    END as sumber,
                    COUNT(*) as total,
                    SUM(CASE WHEN vonis='BERISIK' THEN 1 ELSE 0 END) as berisik,
                    SUM(CASE WHEN vonis='TIDAK BERISIK' THEN 1 ELSE 0 END) as tenang,
                    ROUND(AVG(rms_dbfs), 1) as avg_rms
                FROM deteksi
                GROUP BY 
                    CASE 
                        WHEN sumber LIKE '%Realtime%' THEN 'Realtime WebRTC'
                        WHEN sumber LIKE '%Rekam%' THEN 'Rekam Langsung'
                        WHEN sumber LIKE '%Upload%' OR sumber LIKE '%Unggah%' THEN 'Upload Berkas'
                        ELSE sumber 
                    END
                ORDER BY total DESC
            """).fetchall()

            # 3. Distribusi per Jam (24 jam)
            jam_rows = conn.execute("""
                SELECT strftime('%H', timestamp) as jam,
                       COUNT(*) as total,
                       SUM(CASE WHEN vonis='BERISIK' THEN 1 ELSE 0 END) as berisik
                FROM deteksi
                GROUP BY jam
                ORDER BY jam ASC
            """).fetchall()

            # 4. Distribusi Keyakinan Model
            keyakinan_rows = conn.execute("""
                SELECT keyakinan, COUNT(*) as total
                FROM deteksi
                WHERE keyakinan IS NOT NULL AND keyakinan != ''
                GROUP BY keyakinan
            """).fetchall()

            # 5. Nilai Ekstrem & Rata-rata
            ekstrem = conn.execute("""
                SELECT ROUND(MIN(rms_dbfs), 1) as min_rms,
                       ROUND(MAX(rms_dbfs), 1) as max_rms,
                       ROUND(AVG(rms_dbfs), 1) as avg_rms
                FROM deteksi
                WHERE rms_dbfs IS NOT NULL
            """).fetchone()

            conn.close()
            return {
                "per_metode": [dict(r) for r in metode_rows],
                "per_sumber": [dict(r) for r in sumber_rows],
                "per_jam": [dict(r) for r in jam_rows],
                "keyakinan": [dict(r) for r in keyakinan_rows],
                "ekstrem": dict(ekstrem) if ekstrem else {},
            }
        except Exception:
            return {}


def ambil_daftar_sesi():
    """Ambil seluruh sesi Streamlit beserta status waktu aktifnya."""
    with _lock:
        try:
            conn = _koneksi()
            rows = conn.execute("""
                SELECT sesi_id, mulai, terakhir,
                       CAST((julianday('now', 'localtime') - julianday(terakhir)) * 86400 AS INTEGER) as detik_lalu
                FROM sesi_aktif
                ORDER BY terakhir DESC
            """).fetchall()
            conn.close()
            hasil = []
            for r in rows:
                d = dict(r)
                detik = d.get("detik_lalu", 999)
                d["status"] = "Online" if detik <= 60 else "Idle" if detik <= 300 else "Offline"
                hasil.append(d)
            return hasil
        except Exception:
            return []


def bersihkan_sesi_manual():
    """Hapus semua sesi yang tidak aktif lebih dari 60 detik."""
    hapus_sesi_mati(batas_detik=60)
    return {"status": "ok", "pesan": "Sesi tidak aktif berhasil dibersihkan"}


def ambil_status_sistem():
    """Statistik sistem dan kesehatan database SQLite."""
    with _lock:
        try:
            ukuran_db = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0
            conn = _koneksi()
            total_deteksi = conn.execute("SELECT COUNT(*) FROM deteksi").fetchone()[0]
            total_sesi = conn.execute("SELECT COUNT(*) FROM sesi_aktif").fetchone()[0]
            conn.close()
            return {
                "db_path": DB_PATH,
                "db_size_kb": round(ukuran_db / 1024, 2),
                "total_deteksi": total_deteksi,
                "total_sesi": total_sesi,
                "models": {
                    "Random Forest": "84.3% (300 pohon)",
                    "KNN": "80.8% (k=13, Manhattan)",
                    "SVM": "78.8% (C=2.0, RobustScaler, RBF)",
                },
                "status": "Healthy",
            }
        except Exception as e:
            return {"status": "Error", "error": str(e)}
