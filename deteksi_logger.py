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
    """Ambil 1 baris deteksi terbaru."""
    with _lock:
        try:
            conn = _koneksi()
            row = conn.execute(
                "SELECT * FROM deteksi ORDER BY id DESC LIMIT 1"
            ).fetchone()
            conn.close()
            return dict(row) if row else None
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
            conn.close()
            return {
                "total": total,
                "berisik": berisik,
                "tenang": total - berisik,
                "hari_ini": hari_ini,
                "berisik_hari_ini": berisik_hari_ini,
                "sesi_aktif": sesi,
                "persen_berisik": round(berisik / total * 100, 1) if total else 0,
            }
        except Exception:
            return {}


def ambil_riwayat_terfilter(limit: int = 50, vonis: str = None, metode: str = None, query: str = None):
    """Ambil riwayat deteksi dengan filter dan pencarian untuk log viewer."""
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

            # 2. Distribusi per Sumber
            sumber_rows = conn.execute("""
                SELECT sumber,
                       COUNT(*) as total,
                       SUM(CASE WHEN vonis='BERISIK' THEN 1 ELSE 0 END) as berisik
                FROM deteksi
                GROUP BY sumber
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
