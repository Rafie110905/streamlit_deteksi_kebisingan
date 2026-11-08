"""
admin_panel/main.py
Backend FastAPI & Dashboard Admin Multi-Menu Realtime Deteksi Kebisingan.
Menyediakan Live Monitoring (VU Meter & Line Chart), Deep Analytics, Audit Log,
Manajemen Sesi Pengguna, dan Status Kesehatan Sistem.
"""

import os
import sys
import json
import csv
import io
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import (
    FastAPI, Request, Depends, HTTPException, status,
    WebSocket, WebSocketDisconnect, Query
)
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets

# Tambahkan root project ke sys.path agar bisa import deteksi_logger
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from deteksi_logger import (
    ambil_riwayat, ambil_status_terkini, ambil_data_grafik,
    hitung_statistik, inisialisasi_db, ambil_riwayat_terfilter,
    ambil_analisis_mendalam, ambil_daftar_sesi, bersihkan_sesi_manual,
    ambil_status_sistem
)

# ── Kredensial dari ENV (default: admin / admin123) ──────────────────────────
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "admin123")

# ── Inisialisasi Aplikasi & Keamanan ──────────────────────────────────────────
app = FastAPI(title="Admin Panel Deteksi Kebisingan", docs_url=None, redoc_url=None)
security = HTTPBasic(auto_error=False)

inisialisasi_db()


def verifikasi_admin(
    request: Request,
    credentials: Optional[HTTPBasicCredentials] = Depends(security),
):
    # Cek 1: Token dari parameter URL atau Cookie
    token = request.query_params.get("token") or request.cookies.get("admin_token")
    if token and secrets.compare_digest(token.encode("utf-8"), ADMIN_PASS.encode("utf-8")):
        return ADMIN_USER

    # Cek 2: Kredensial HTTP Basic
    if credentials:
        ok_user = secrets.compare_digest(credentials.username.encode("utf-8"), ADMIN_USER.encode("utf-8"))
        ok_pass = secrets.compare_digest(credentials.password.encode("utf-8"), ADMIN_PASS.encode("utf-8"))
        if ok_user and ok_pass:
            return credentials.username

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Kredensial salah atau token tidak valid",
        headers={"WWW-Authenticate": "Basic"},
    )


@app.get("/embed", response_class=HTMLResponse)
def dashboard_embed(request: Request, token: str = Query("")):
    """Tampilan embed dashboard admin dengan otentikasi token untuk Streamlit."""
    if not (token and secrets.compare_digest(token.encode("utf-8"), ADMIN_PASS.encode("utf-8"))):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token otentikasi salah atau tidak ada",
            headers={"WWW-Authenticate": "Basic"},
        )
    resp = HTMLResponse(content=DASHBOARD_HTML)
    resp.set_cookie(key="admin_token", value=ADMIN_PASS, httponly=True, samesite="lax")
    return resp


# ── REST API Endpoints ────────────────────────────────────────────────────────
@app.get("/api/status", dependencies=[Depends(verifikasi_admin)])
def api_status():
    """Status deteksi terkini + statistik ringkasan."""
    return JSONResponse({
        "terkini": ambil_status_terkini(),
        "statistik": hitung_statistik(),
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


@app.get("/api/riwayat", dependencies=[Depends(verifikasi_admin)])
def api_riwayat(
    limit: int = Query(50, ge=1, le=500),
    vonis: str = Query("Semua"),
    metode: str = Query("Semua"),
    sumber: str = Query("Semua"),
    query: str = Query("")
):
    """Riwayat deteksi dengan filter vonis, metode, sumber, dan pencarian."""
    data = ambil_riwayat_terfilter(limit=limit, vonis=vonis, metode=metode, query=query, sumber=sumber)
    return JSONResponse({"data": data, "total": len(data)})


@app.get("/api/grafik", dependencies=[Depends(verifikasi_admin)])
def api_grafik(menit: int = Query(10, ge=1, le=120)):
    """Data grafik level suara N menit terakhir."""
    return JSONResponse({"data": ambil_data_grafik(menit=menit)})


@app.get("/api/analisis", dependencies=[Depends(verifikasi_admin)])
def api_analisis():
    """Agregasi analitik mendalam (breakdown metode, sumber, jam, keyakinan)."""
    return JSONResponse({"data": ambil_analisis_mendalam()})


@app.get("/api/sesi", dependencies=[Depends(verifikasi_admin)])
def api_sesi():
    """Daftar sesi Streamlit aktif beserta status online/offline."""
    return JSONResponse({"data": ambil_daftar_sesi()})


@app.post("/api/sesi/bersihkan", dependencies=[Depends(verifikasi_admin)])
def api_bersihkan_sesi():
    """Hapus sesi yang tidak aktif lebih dari 60 detik."""
    return JSONResponse(bersihkan_sesi_manual())


@app.get("/api/sistem", dependencies=[Depends(verifikasi_admin)])
def api_sistem():
    """Status kesehatan sistem dan basis data SQLite."""
    return JSONResponse({"data": ambil_status_sistem()})


@app.get("/api/ekspor/csv", dependencies=[Depends(verifikasi_admin)])
def api_ekspor_csv():
    """Ekspor seluruh data riwayat deteksi ke dalam format CSV."""
    semua_data = ambil_riwayat(limit=5000)
    output = io.StringIO()
    penulis = csv.writer(output)
    penulis.writerow(["ID", "Timestamp", "Sumber", "Metode", "Vonis", "P_Berisik", "RMS_dBFS", "Nama_File", "Keyakinan", "Sesi_ID"])
    for r in semua_data:
        penulis.writerow([
            r.get("id"), r.get("timestamp"), r.get("sumber"), r.get("metode"),
            r.get("vonis"), r.get("p_berisik"), r.get("rms_dbfs"),
            r.get("nama_file"), r.get("keyakinan"), r.get("sesi_id")
        ])
    nama_file = f"riwayat_kebisingan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={nama_file}"}
    )


# ── WebSocket Realtime Streaming ─────────────────────────────────────────────
@app.websocket("/ws/live")
async def websocket_live_endpoint(websocket: WebSocket):
    """Kanal streaming WebSocket berkecepatan tinggi untuk push data realtime."""
    await websocket.accept()
    try:
        while True:
            payload = {
                "terkini": ambil_status_terkini(),
                "statistik": hitung_statistik(),
                "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(1.0)  # Push setiap 1 detik
    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass


# ── Dashboard HTML Multi-Menu Realtime ───────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Admin Dashboard — Smart Noise Intelligence</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg-dark: #0b1120; --bg-card: #131d33; --bg-hover: #1e293b;
    --border: #24324f; --text-main: #f8fafc; --text-sub: #94a3b8;
    --merah: #ef4444; --hijau: #10b981; --kuning: #f59e0b;
    --biru: #3b82f6; --ungu: #8b5cf6; --cyan: #06b6d4;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg-dark); color: var(--text-main); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; display: flex; min-height: 100vh; }
  
  /* Sidebar */
  aside { width: 260px; background: #070d1a; border-right: 1px solid var(--border); display: flex; flex-direction: column; flex-shrink: 0; }
  .brand { padding: 22px 20px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 12px; }
  .brand-icon { width: 38px; height: 38px; border-radius: 10px; background: linear-gradient(135deg, #3b82f6, #8b5cf6); display: flex; align-items: center; justify-content: center; font-size: 20px; }
  .brand-text h2 { font-size: 15px; font-weight: 800; letter-spacing: .5px; }
  .brand-text p { font-size: 11px; color: var(--text-sub); }
  
  .nav-menu { padding: 18px 12px; flex: 1; display: flex; flex-direction: column; gap: 6px; }
  .nav-btn { display: flex; align-items: center; gap: 12px; width: 100%; padding: 12px 14px; border: none; background: transparent; color: var(--text-sub); font-size: 13px; font-weight: 600; border-radius: 10px; cursor: pointer; text-align: left; transition: all .2s; }
  .nav-btn:hover { background: var(--bg-hover); color: var(--text-main); }
  .nav-btn.active { background: #1e293b; color: var(--biru); box-shadow: inset 3px 0 0 var(--biru); }
  .nav-btn .icon { font-size: 16px; width: 20px; text-align: center; }
  
  .sidebar-footer { padding: 16px 20px; border-top: 1px solid var(--border); font-size: 11px; color: var(--text-sub); display: flex; flex-direction: column; gap: 6px; }
  .status-pill { display: inline-flex; align-items: center; gap: 6px; font-weight: 700; color: var(--hijau); font-size: 11px; }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--hijau); animation: blink 1.4s infinite; }
  @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: .3; } }
  
  /* Main Content */
  main { flex: 1; display: flex; flex-direction: column; overflow-y: auto; height: 100vh; }
  .topbar { background: #070d1a; border-bottom: 1px solid var(--border); padding: 16px 32px; display: flex; align-items: center; justify-content: space-between; position: sticky; top: 0; z-index: 10; }
  .topbar-title { font-size: 18px; font-weight: 800; }
  .topbar-actions { display: flex; align-items: center; gap: 14px; }
  .sync-badge { font-size: 11px; background: rgba(59,130,246,.15); color: var(--biru); border: 1px solid rgba(59,130,246,.3); padding: 4px 10px; border-radius: 999px; font-weight: 700; }
  .btn-action { background: var(--bg-card); border: 1px solid var(--border); color: var(--text-main); font-size: 12px; font-weight: 600; padding: 7px 14px; border-radius: 8px; cursor: pointer; transition: all .2s; text-decoration: none; display: inline-flex; align-items: center; gap: 6px; }
  .btn-action:hover { background: var(--border); }
  
  .content-area { padding: 28px 32px; }
  .tab-pane { display: none; }
  .tab-pane.active { display: block; animation: fadeIn .25s ease-out; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
  
  /* Kartu & Grid */
  .grid-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .card-metric { background: var(--bg-card); border: 1px solid var(--border); border-radius: 14px; padding: 20px; transition: transform .2s; }
  .card-metric:hover { transform: translateY(-2px); }
  .card-metric .lbl { font-size: 11px; color: var(--text-sub); text-transform: uppercase; letter-spacing: .8px; font-weight: 700; }
  .card-metric .val { font-size: 34px; font-weight: 900; margin-top: 6px; }
  .card-metric .sub { font-size: 11px; color: var(--text-sub); margin-top: 4px; }
  
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }
  @media(max-width: 1024px){ .grid-2 { grid-template-columns: 1fr; } }
  
  .card-panel { background: var(--bg-card); border: 1px solid var(--border); border-radius: 16px; padding: 22px; margin-bottom: 24px; }
  .card-panel-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 18px; }
  .card-panel-header h3 { font-size: 14px; font-weight: 800; text-transform: uppercase; letter-spacing: .8px; color: var(--text-sub); display: flex; align-items: center; gap: 8px; }
  
  /* Live Status Hero */
  .hero-status { border-radius: 16px; padding: 28px 32px; margin-bottom: 24px; border: 1px solid var(--border); border-left-width: 10px; display: flex; align-items: center; justify-content: space-between; gap: 24px; transition: all .4s; }
  .hero-status.berisik { border-left-color: var(--merah); background: linear-gradient(90deg, rgba(239,68,68,.12) 0%, rgba(19,29,51,0) 80%); }
  .hero-status.tenang  { border-left-color: var(--hijau); background: linear-gradient(90deg, rgba(16,185,129,.12) 0%, rgba(19,29,51,0) 80%); }
  .hero-status.kosong  { border-left-color: var(--border); background: var(--bg-card); }
  .hero-title { font-size: 44px; font-weight: 900; letter-spacing: -.5px; line-height: 1.1; }
  .hero-title.berisik { color: var(--merah); text-shadow: 0 0 20px rgba(239,68,68,.3); }
  .hero-title.tenang  { color: var(--hijau); text-shadow: 0 0 20px rgba(16,185,129,.3); }
  
  /* VU Meter (Sound Level Bar) */
  .vu-container { margin-top: 14px; width: 100%; max-width: 480px; }
  .vu-bar-wrap { position: relative; height: 16px; background: #070d1a; border-radius: 8px; border: 1px solid var(--border); overflow: hidden; }
  .vu-bar-fill { height: 100%; width: 0%; border-radius: 8px; transition: width .3s ease-out, background .3s; }
  .vu-labels { display: flex; justify-content: space-between; font-size: 10px; color: var(--text-sub); margin-top: 4px; }
  
  /* Tabel Riwayat */
  .table-tools { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 16px; align-items: center; justify-content: space-between; }
  .filter-group { display: flex; align-items: center; gap: 8px; }
  .input-control { background: #070d1a; border: 1px solid var(--border); color: var(--text-main); font-size: 12px; padding: 7px 12px; border-radius: 8px; outline: none; }
  .input-control:focus { border-color: var(--biru); }
  
  .table-responsive { overflow-x: auto; border: 1px solid var(--border); border-radius: 12px; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; text-align: left; }
  th { background: #0e1628; padding: 12px 14px; color: var(--text-sub); font-size: 11px; text-transform: uppercase; letter-spacing: .6px; border-bottom: 1px solid var(--border); font-weight: 700; }
  td { padding: 12px 14px; border-bottom: 1px solid rgba(36,50,79,.6); vertical-align: middle; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: rgba(255,255,255,.02); }
  
  .badge { display: inline-flex; align-items: center; gap: 5px; padding: 3px 9px; border-radius: 999px; font-weight: 700; font-size: 11px; }
  .badge-berisik { background: rgba(239,68,68,.18); color: var(--merah); border: 1px solid rgba(239,68,68,.3); }
  .badge-tenang  { background: rgba(16,185,129,.18); color: var(--hijau); border: 1px solid rgba(16,185,129,.3); }
  .badge-online  { background: rgba(16,185,129,.18); color: var(--hijau); border: 1px solid rgba(16,185,129,.3); }
  .badge-idle    { background: rgba(245,158,11,.18); color: var(--kuning); border: 1px solid rgba(245,158,11,.3); }
  .badge-offline { background: rgba(148,163,184,.15); color: var(--text-sub); border: 1px solid rgba(148,163,184,.3); }
  .badge-src-realtime { background: rgba(6,182,212,.18); color: var(--cyan); border: 1px solid rgba(6,182,212,.35); }
  .badge-src-rekam    { background: rgba(139,92,246,.18); color: var(--ungu); border: 1px solid rgba(139,92,246,.35); }
  .badge-src-upload   { background: rgba(59,130,246,.18); color: var(--biru); border: 1px solid rgba(59,130,246,.35); }
  
  /* Toggle Switch */
  .switch { position: relative; display: inline-block; width: 38px; height: 20px; }
  .switch input { opacity: 0; width: 0; height: 0; }
  .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: var(--border); transition: .3s; border-radius: 20px; }
  .slider:before { position: absolute; content: ""; height: 14px; width: 14px; left: 3px; bottom: 3px; background-color: white; transition: .3s; border-radius: 50%; }
  input:checked + .slider { background-color: var(--biru); }
  input:checked + .slider:before { transform: translateX(18px); }
</style>
</head>
<body>

<!-- SIDEBAR -->
<aside>
  <div class="brand">
    <div class="brand-icon">🔊</div>
    <div class="brand-text">
      <h2>NOISE ADMIN</h2>
      <p>Room Acoustic Intel</p>
    </div>
  </div>
  
  <nav class="nav-menu">
    <button class="nav-btn active" onclick="bukaTab('tab-live', this)">
      <span class="icon">📡</span> Monitoring Realtime
    </button>
    <button class="nav-btn" onclick="bukaTab('tab-analisis', this)">
      <span class="icon">📊</span> Analisis & Statistik
    </button>
    <button class="nav-btn" onclick="bukaTab('tab-riwayat', this)">
      <span class="icon">📋</span> Riwayat & Audit Log
    </button>
    <button class="nav-btn" onclick="bukaTab('tab-sesi', this)">
      <span class="icon">👥</span> Sesi Pengguna
    </button>
    <button class="nav-btn" onclick="bukaTab('tab-sistem', this)">
      <span class="icon">⚙️</span> Sistem & Model
    </button>
  </nav>
  
  <div class="sidebar-footer">
    <div class="status-pill">
      <span class="status-dot"></span> <span id="lbl-koneksi-sidebar">WebSocket Live</span>
    </div>
    <div style="font-size:10px;color:#64748b" id="txt-uptime-sidebar">Server Aktif: Port 8000</div>
  </div>
</aside>

<!-- MAIN CONTENT -->
<main>
  <div class="topbar">
    <div class="topbar-title" id="page-title">📡 Monitoring Realtime</div>
    <div class="topbar-actions">
      <div style="display:flex;align-items:center;gap:8px;font-size:12px;color:var(--text-sub)">
        <span>Alarm Suara:</span>
        <label class="switch">
          <input type="checkbox" id="chk-alarm">
          <span class="slider"></span>
        </label>
      </div>
      <span class="sync-badge" id="badge-sync">1s SYNC</span>
      <span style="font-size:12px;color:var(--text-sub)" id="txt-last-sync">Memuat…</span>
    </div>
  </div>
  
  <div class="content-area">

    <!-- ── TAB 1: LIVE MONITORING ─────────────────────────────────────────── -->
    <div id="tab-live" class="tab-pane active">
      
      <!-- Status Hero Card -->
      <div class="hero-status kosong" id="hero-card">
        <div>
          <div style="font-size:12px;letter-spacing:2px;color:var(--text-sub);margin-bottom:6px;font-weight:700">STATUS AKUSTIK TERKINI</div>
          <div class="hero-title" id="hero-vonis">—</div>
          <div style="font-size:13px;color:var(--text-sub);margin-top:6px" id="hero-meta">Menghubungkan ke sensor audio…</div>
          
          <!-- Sound Level Meter (VU Meter) -->
          <div class="vu-container">
            <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:4px">
              <span>Digital VU Meter</span>
              <b id="vu-val-text">— dBFS</b>
            </div>
            <div class="vu-bar-wrap">
              <div class="vu-bar-fill" id="vu-bar" style="width:0%;background:var(--hijau)"></div>
            </div>
            <div class="vu-labels">
              <span>-60 dB (Sunyi)</span>
              <span>-30 dB (Wajar)</span>
              <span>-20 dB (Batas)</span>
              <span>0 dB (Keras)</span>
            </div>
          </div>
        </div>
        
        <div style="text-align:right;display:flex;flex-direction:column;align-items:flex-end;gap:10px">
          <div class="status-dot" id="hero-dot" style="width:22px;height:22px;background:#334155"></div>
          <div style="background:#070d1a;padding:10px 14px;border-radius:10px;border:1px solid var(--border);text-align:right">
            <div style="font-size:10px;color:var(--text-sub)">KEYAKINAN MODEL</div>
            <div style="font-size:14px;font-weight:800;color:var(--text-main)" id="hero-keyakinan">—</div>
          </div>
        </div>
      </div>

      <!-- Quick Summary Metrics -->
      <div class="grid-metrics">
        <div class="card-metric">
          <div class="lbl">Total Deteksi</div>
          <div class="val" style="color:var(--biru)" id="m-total">—</div>
          <div class="sub">Keseluruhan sesi</div>
        </div>
        <div class="card-metric">
          <div class="lbl">Vonis Berisik</div>
          <div class="val" style="color:var(--merah)" id="m-berisik">—</div>
          <div class="sub" id="m-persen">— % rasio</div>
        </div>
        <div class="card-metric">
          <div class="lbl">Vonis Tenang</div>
          <div class="val" style="color:var(--hijau)" id="m-tenang">—</div>
          <div class="sub">Kondisi kondusif</div>
        </div>
        <div class="card-metric">
          <div class="lbl">Deteksi Hari Ini</div>
          <div class="val" style="color:var(--kuning)" id="m-hari-ini">—</div>
          <div class="sub" id="m-berisik-hari-ini">— berisik</div>
        </div>
        <div class="card-metric">
          <div class="lbl">Klien Aktif</div>
          <div class="val" style="color:var(--cyan)" id="m-sesi">—</div>
          <div class="sub">Sesi Streamlit aktif</div>
        </div>
      </div>

      <!-- Breakdown Rincian Berdasarkan Sumber Audio -->
      <div style="margin-bottom:12px;font-size:11px;font-weight:700;color:var(--text-sub);text-transform:uppercase;letter-spacing:.8px">
        DISTRIBUSI DATA BERDASARKAN SUMBER AUDIO
      </div>
      <div class="grid-metrics" style="grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin-bottom: 24px;">
        <div class="card-metric" style="border-left: 4px solid var(--cyan)">
          <div class="lbl">📡 Realtime WebRTC</div>
          <div class="val" style="color:var(--cyan)" id="m-src-realtime">—</div>
          <div class="sub">Streaming mikrofon browser langsung</div>
        </div>
        <div class="card-metric" style="border-left: 4px solid var(--ungu)">
          <div class="lbl">🎙️ Rekam Langsung</div>
          <div class="val" style="color:var(--ungu)" id="m-src-rekam">—</div>
          <div class="sub">Audio input rekaman suara pengguna</div>
        </div>
        <div class="card-metric" style="border-left: 4px solid var(--biru)">
          <div class="lbl">📁 Upload Berkas</div>
          <div class="val" style="color:var(--biru)" id="m-src-upload">—</div>
          <div class="sub">Berkas MP3/WAV/FLAC diunggah</div>
        </div>
      </div>

      <!-- Live Line Chart -->
      <div class="card-panel">
        <div class="card-panel-header">
          <h3>📈 Aliran Level Suara (dBFS) — 10 Menit Terakhir</h3>
          <span style="font-size:11px;color:var(--text-sub)">Update otomatis setiap detik</span>
        </div>
        <div style="position:relative;height:240px">
          <canvas id="chart-live-rms"></canvas>
        </div>
      </div>

    </div>

    <!-- ── TAB 2: ANALISIS & STATISTIK ────────────────────────────────────── -->
    <div id="tab-analisis" class="tab-pane">
      
      <div class="grid-metrics">
        <div class="card-metric">
          <div class="lbl">RMS Terkeras (Peak)</div>
          <div class="val" style="color:var(--merah)" id="an-max-rms">— dBFS</div>
          <div class="sub">Puncak energi sinyal</div>
        </div>
        <div class="card-metric">
          <div class="lbl">RMS Rata-Rata</div>
          <div class="val" style="color:var(--biru)" id="an-avg-rms">— dBFS</div>
          <div class="sub">Tingkat kebisingan rata-rata</div>
        </div>
        <div class="card-metric">
          <div class="lbl">RMS Paling Hening</div>
          <div class="val" style="color:var(--hijau)" id="an-min-rms">— dBFS</div>
          <div class="sub">Level kebisingan dasar (floor)</div>
        </div>
      </div>

      <div class="grid-2">
        <!-- Chart 1: Kebisingan per Jam -->
        <div class="card-panel">
          <div class="card-panel-header">
            <h3>⏰ Pola Kebisingan Berdasarkan Jam (24 Jam)</h3>
          </div>
          <div style="position:relative;height:260px">
            <canvas id="chart-hourly"></canvas>
          </div>
        </div>

        <!-- Chart 2: Komparasi Metode -->
        <div class="card-panel">
          <div class="card-panel-header">
            <h3>🤖 Komparasi Metode Klasifikasi</h3>
          </div>
          <div style="position:relative;height:260px">
            <canvas id="chart-metode"></canvas>
          </div>
        </div>
      </div>

      <div class="grid-2">
        <!-- Chart 3: Distribusi Keyakinan -->
        <div class="card-panel">
          <div class="card-panel-header">
            <h3>🎯 Distribusi Keyakinan Model AI</h3>
          </div>
          <div style="position:relative;height:240px">
            <canvas id="chart-keyakinan"></canvas>
          </div>
        </div>

        <!-- Chart 4: Proporsi Sumber -->
        <div class="card-panel">
          <div class="card-panel-header">
            <h3>🎙️ Komparasi Kebisingan per Sumber Audio</h3>
          </div>
          <div style="position:relative;height:240px">
            <canvas id="chart-sumber"></canvas>
          </div>
        </div>
      </div>

      <!-- Tabel Komparasi Karakteristik Sumber Audio -->
      <div class="card-panel">
        <div class="card-panel-header">
          <h3>📊 Karakteristik Akustik Berdasarkan Sumber Audio</h3>
        </div>
        <div class="table-responsive">
          <table>
            <thead>
              <tr>
                <th>Sumber Audio</th>
                <th>Total Pengujian</th>
                <th>Vonis Berisik</th>
                <th>Vonis Tenang</th>
                <th>Rasio Berisik (%)</th>
                <th>Rata-rata RMS (dBFS)</th>
              </tr>
            </thead>
            <tbody id="tbody-analisis-sumber">
              <tr><td colspan="6" style="text-align:center;padding:16px;color:var(--text-sub)">Memuat data sumber audio…</td></tr>
            </tbody>
          </table>
        </div>
      </div>

    </div>

    <!-- ── TAB 3: RIWAYAT & AUDIT LOG ────────────────────────────────────── -->
    <div id="tab-riwayat" class="tab-pane">
      <div class="card-panel">
        
        <div class="table-tools">
          <div class="filter-group">
            <label style="font-size:11px;color:var(--text-sub);font-weight:700">SUMBER:</label>
            <select class="input-control" id="f-sumber" onchange="muatRiwayat()">
              <option value="Semua">Semua Sumber</option>
              <option value="Realtime WebRTC">📡 Realtime WebRTC</option>
              <option value="Rekam Langsung">🎙️ Rekam Langsung</option>
              <option value="Upload Berkas">📁 Upload Berkas</option>
            </select>

            <label style="font-size:11px;color:var(--text-sub);font-weight:700;margin-left:6px">VONIS:</label>
            <select class="input-control" id="f-vonis" onchange="muatRiwayat()">
              <option value="Semua">Semua Vonis</option>
              <option value="BERISIK">Berisik Saja</option>
              <option value="TIDAK BERISIK">Tenang Saja</option>
            </select>

            <label style="font-size:11px;color:var(--text-sub);font-weight:700;margin-left:6px">METODE:</label>
            <select class="input-control" id="f-metode" onchange="muatRiwayat()">
              <option value="Semua">Semua Metode</option>
              <option value="Random Forest">Random Forest</option>
              <option value="KNN">KNN</option>
              <option value="SVM">SVM</option>
              <option value="RMS">Ambang RMS</option>
            </select>

            <label style="font-size:11px;color:var(--text-sub);font-weight:700;margin-left:6px">BARIS:</label>
            <select class="input-control" id="f-limit" onchange="muatRiwayat()">
              <option value="25">25</option>
              <option value="50" selected>50</option>
              <option value="100">100</option>
              <option value="200">200</option>
            </select>
          </div>

          <div class="filter-group">
            <input type="text" class="input-control" id="f-query" placeholder="🔍 Cari nama file / sesi..." oninput="debounceRiwayat()">
            <a href="/api/ekspor/csv" id="btn-csv-export" class="btn-action">📥 Unduh CSV</a>
          </div>
        </div>

        <div class="table-responsive">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Waktu</th>
                <th>Sumber</th>
                <th>Metode</th>
                <th>Vonis</th>
                <th>P(Berisik)</th>
                <th>Level RMS</th>
                <th>Keyakinan</th>
                <th>Nama File / Info</th>
                <th>Sesi ID</th>
              </tr>
            </thead>
            <tbody id="tbody-riwayat">
              <tr><td colspan="10" style="text-align:center;padding:24px;color:var(--text-sub)">Memuat riwayat deteksi…</td></tr>
            </tbody>
          </table>
        </div>

      </div>
    </div>

    <!-- ── TAB 4: SESI PENGGUNA ───────────────────────────────────────────── -->
    <div id="tab-sesi" class="tab-pane">
      <div class="card-panel">
        <div class="card-panel-header">
          <h3>👥 Daftar Sesi Klien Streamlit</h3>
          <button class="btn-action" onclick="bersihkanSesiMati()">🧹 Bersihkan Sesi Tidak Aktif</button>
        </div>
        <p style="font-size:12px;color:var(--text-sub);margin-bottom:16px">
          Sistem secara otomatis mendeteksi sesi browser pengguna yang terhubung. Klien dengan aktivitas kurang dari 60 detik ditandai sebagai Online.
        </p>

        <div class="table-responsive">
          <table>
            <thead>
              <tr>
                <th>Status</th>
                <th>Session ID</th>
                <th>Waktu Mulai</th>
                <th>Aktivitas Terakhir</th>
                <th>Terakhir Aktif</th>
              </tr>
            </thead>
            <tbody id="tbody-sesi">
              <tr><td colspan="5" style="text-align:center;padding:24px;color:var(--text-sub)">Memuat sesi aktif…</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- ── TAB 5: SISTEM & MODEL ─────────────────────────────────────────── -->
    <div id="tab-sistem" class="tab-pane">
      
      <div class="grid-2">
        <!-- Info Model -->
        <div class="card-panel">
          <div class="card-panel-header">
            <h3>🤖 Model Machine Learning Aktif</h3>
          </div>
          <div style="display:flex;flex-direction:column;gap:12px;font-size:13px">
            <div style="background:#070d1a;padding:14px;border-radius:10px;border:1px solid var(--border)">
              <div style="font-weight:800;color:var(--biru)">🌲 Random Forest Classifier</div>
              <div style="color:var(--text-sub);font-size:11px;margin-top:2px">Akurasi CV: <b>84.34%</b> · 300 Trees · 25 Fitur Akustik</div>
            </div>
            <div style="background:#070d1a;padding:14px;border-radius:10px;border:1px solid var(--border)">
              <div style="font-weight:800;color:var(--hijau)">📍 K-Nearest Neighbors (KNN)</div>
              <div style="color:var(--text-sub);font-size:11px;margin-top:2px">Akurasi CV: <b>80.81%</b> · k=13 · Jarak Manhattan (L1) · Distance Weighted</div>
            </div>
            <div style="background:#070d1a;padding:14px;border-radius:10px;border:1px solid var(--border)">
              <div style="font-weight:800;color:var(--ungu)">🎯 Support Vector Machine (SVM)</div>
              <div style="color:var(--text-sub);font-size:11px;margin-top:2px">Akurasi CV: <b>78.79%</b> · C=2.0 · RBF Kernel · RobustScaler (IQR)</div>
            </div>
          </div>
        </div>

        <!-- Info Database & Server -->
        <div class="card-panel">
          <div class="card-panel-header">
            <h3>🗄️ Kesehatan Database & Server</h3>
          </div>
          <div style="display:flex;flex-direction:column;gap:12px;font-size:13px">
            <div style="display:flex;justify-content:space-between;border-bottom:1px solid var(--border);padding-bottom:8px">
              <span style="color:var(--text-sub)">Status Server:</span>
              <span style="color:var(--hijau);font-weight:700">🟢 Running (Healthy)</span>
            </div>
            <div style="display:flex;justify-content:space-between;border-bottom:1px solid var(--border);padding-bottom:8px">
              <span style="color:var(--text-sub)">Database SQLite:</span>
              <span id="sys-db-path">—</span>
            </div>
            <div style="display:flex;justify-content:space-between;border-bottom:1px solid var(--border);padding-bottom:8px">
              <span style="color:var(--text-sub)">Ukuran Basis Data:</span>
              <span id="sys-db-size">— KB</span>
            </div>
            <div style="display:flex;justify-content:space-between;border-bottom:1px solid var(--border);padding-bottom:8px">
              <span style="color:var(--text-sub)">Total Rekaman Log:</span>
              <span id="sys-total-rows">— baris</span>
            </div>
            <div style="display:flex;justify-content:space-between">
              <span style="color:var(--text-sub)">Protokol Realtime:</span>
              <span style="color:var(--cyan);font-weight:700">WebSocket + HTTP Fallback</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Standar Kebisingan -->
      <div class="card-panel">
        <div class="card-panel-header">
          <h3>📜 Regulasi Ambang Batas Kebisingan</h3>
        </div>
        <div style="font-size:13px;line-height:1.6;color:var(--text-sub)">
          Berdasarkan <b>Permenaker No. 5 Tahun 2018</b> dan <b>Kepmen LH No. 48 Tahun 1996</b>:
          <ul style="margin-left:20px;margin-top:8px">
            <li><b>Zona Sunyi / Tenang (Di bawah -30 dBFS / < 55 dBA)</b>: Sesuai untuk ruang kerja kantor, ruang belajar, dan perpustakaan.</li>
            <li><b>Zona Waspada (-30 hingga -20 dBFS / 55 - 70 dBA)</b>: Batas percakapan normal; mulai mengganggu fokus konsentrasi tinggi.</li>
            <li><b>Zona Berisik (Di atas -20 dBFS / > 85 dBA)</b>: Wajib tindakan mitigasi kebisingan atau pemindahan sumber kegaduhan.</li>
          </ul>
        </div>
      </div>

    </div>

  </div>
</main>

<script>
// ── Web Audio API Beeper (Alarm Realtime) ───────────────────────────────────
let audioCtx = null;
function bunyikanBuzzer() {
  const chk = document.getElementById('chk-alarm');
  if (!chk || !chk.checked) return;
  try {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(880, audioCtx.currentTime); // 880 Hz
    gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.15); // bunyi 150ms
  } catch(e) {}
}

// ── Tab Navigation ─────────────────────────────────────────────────────────
function bukaTab(tabId, btn) {
  document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
  
  const target = document.getElementById(tabId);
  if (target) target.classList.add('active');
  if (btn) btn.classList.add('active');

  const titleMap = {
    'tab-live': '📡 Monitoring Realtime',
    'tab-analisis': '📊 Analisis & Statistik Mendalam',
    'tab-riwayat': '📋 Riwayat & Audit Log',
    'tab-sesi': '👥 Sesi Pengguna Streamlit',
    'tab-sistem': '⚙️ Sistem & Performa Model'
  };
  document.getElementById('page-title').textContent = titleMap[tabId] || 'Admin Dashboard';

  if (tabId === 'tab-analisis') muatAnalisis();
  if (tabId === 'tab-riwayat') muatRiwayat();
  if (tabId === 'tab-sesi') muatSesi();
  if (tabId === 'tab-sistem') muatSistem();
}

// ── Chart.js Configurations ────────────────────────────────────────────────
const COLOR_RED = '#ef4444', COLOR_GREEN = '#10b981', COLOR_BLUE = '#3b82f6', COLOR_YELLOW = '#f59e0b', COLOR_PURPLE = '#8b5cf6';

// 1. Live RMS Chart
const ctxLive = document.getElementById('chart-live-rms').getContext('2d');
const chartLiveRms = new Chart(ctxLive, {
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      label: 'Level RMS (dBFS)',
      data: [],
      borderColor: COLOR_BLUE,
      backgroundColor: 'rgba(59,130,246,0.12)',
      tension: 0.35, fill: true, borderWidth: 2, pointRadius: 3
    }, {
      label: 'Batas Berisik (-20 dBFS)',
      data: [],
      borderColor: 'rgba(239,68,68,0.7)',
      borderDash: [5, 4], pointRadius: 0, borderWidth: 1.5, fill: false
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: false, animation: { duration: 300 },
    scales: {
      x: { ticks: { color: '#94a3b8', maxTicksLimit: 10 }, grid: { color: '#17223b' } },
      y: { ticks: { color: '#94a3b8' }, grid: { color: '#24324f' }, title: { display: true, text: 'dBFS', color: '#94a3b8' } }
    },
    plugins: { legend: { labels: { color: '#f8fafc', font: { size: 11 } } } }
  }
});

// 2. Hourly Chart
let chartHourly = null;
function initChartHourly(labels, berisik, total) {
  const ctx = document.getElementById('chart-hourly').getContext('2d');
  if (chartHourly) chartHourly.destroy();
  chartHourly = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [
        { label: 'Berisik', data: berisik, backgroundColor: 'rgba(239,68,68,0.75)', borderRadius: 4 },
        { label: 'Total', data: total, backgroundColor: 'rgba(59,130,246,0.4)', borderRadius: 4 }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { x: { ticks: { color: '#94a3b8' } }, y: { ticks: { color: '#94a3b8' }, grid: { color: '#24324f' } } },
      plugins: { legend: { labels: { color: '#f8fafc' } } }
    }
  });
}

// 3. Metode Chart
let chartMetode = null;
function initChartMetode(labels, berisik, tenang) {
  const ctx = document.getElementById('chart-metode').getContext('2d');
  if (chartMetode) chartMetode.destroy();
  chartMetode = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [
        { label: 'Berisik', data: berisik, backgroundColor: COLOR_RED, borderRadius: 4 },
        { label: 'Tenang', data: tenang, backgroundColor: COLOR_GREEN, borderRadius: 4 }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { x: { ticks: { color: '#94a3b8' } }, y: { ticks: { color: '#94a3b8' }, grid: { color: '#24324f' } } },
      plugins: { legend: { labels: { color: '#f8fafc' } } }
    }
  });
}

// 4. Keyakinan Chart
let chartKeyakinan = null;
function initChartKeyakinan(labels, data) {
  const ctx = document.getElementById('chart-keyakinan').getContext('2d');
  if (chartKeyakinan) chartKeyakinan.destroy();
  chartKeyakinan = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: labels,
      datasets: [{ data: data, backgroundColor: [COLOR_GREEN, COLOR_BLUE, COLOR_YELLOW, COLOR_RED], borderWidth: 2, borderColor: '#131d33' }]
    },
    options: { responsive: true, maintainAspectRatio: false, cutout: '65%', plugins: { legend: { position: 'bottom', labels: { color: '#f8fafc' } } } }
  });
}

// 5. Sumber Chart
let chartSumber = null;
function initChartSumber(labels, berisik, tenang) {
  const ctx = document.getElementById('chart-sumber').getContext('2d');
  if (chartSumber) chartSumber.destroy();
  chartSumber = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [
        { label: 'Berisik', data: berisik, backgroundColor: COLOR_RED, borderRadius: 4 },
        { label: 'Tenang', data: tenang, backgroundColor: COLOR_GREEN, borderRadius: 4 }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { x: { ticks: { color: '#94a3b8' } }, y: { ticks: { color: '#94a3b8' }, grid: { color: '#24324f' } } },
      plugins: { legend: { labels: { color: '#f8fafc' } } }
    }
  });
}

// ── Helpers ────────────────────────────────────────────────────────────────
const urlParams = new URLSearchParams(window.location.search);
const adminToken = urlParams.get('token') || '';

function apiUrl(endpoint) {
  if (!adminToken) return endpoint;
  const sep = endpoint.includes('?') ? '&' : '?';
  return `${endpoint}${sep}token=${encodeURIComponent(adminToken)}`;
}

function setText(id, text) { const el = document.getElementById(id); if (el) el.textContent = text; }
function fmt(val, dec=1) { return (val === null || val === undefined) ? '—' : parseFloat(val).toFixed(dec); }

function formatSumberBadge(sumber) {
  if (!sumber) return '—';
  if (sumber.includes('Realtime')) {
    return '<span class="badge badge-src-realtime">📡 Realtime WebRTC</span>';
  } else if (sumber.includes('Rekam')) {
    return '<span class="badge badge-src-rekam">🎙️ Rekam Langsung</span>';
  } else if (sumber.includes('Upload') || sumber.includes('Unggah')) {
    return '<span class="badge badge-src-upload">📁 Upload Berkas</span>';
  }
  return `<span class="badge">${sumber}</span>`;
}

// ── Update Live Monitoring UI ──────────────────────────────────────────────
let lastVonis = null;

function applyLivePayload(d) {
  const st = d.terkini;
  const stat = d.statistik;

  if (st) {
    const isBerisik = st.vonis === 'BERISIK';
    const heroCard = document.getElementById('hero-card');
    const heroVonis = document.getElementById('hero-vonis');
    const heroMeta = document.getElementById('hero-meta');
    const heroDot = document.getElementById('hero-dot');
    const heroKey = document.getElementById('hero-keyakinan');

    heroCard.className = 'hero-status ' + (isBerisik ? 'berisik' : 'tenang');
    heroVonis.className = 'hero-title ' + (isBerisik ? 'berisik' : 'tenang');
    heroVonis.textContent = st.vonis;
    heroDot.style.background = isBerisik ? 'var(--merah)' : 'var(--hijau)';
    heroDot.style.boxShadow = isBerisik ? '0 0 16px rgba(239,68,68,0.8)' : '0 0 12px rgba(16,185,129,0.5)';
    heroMeta.textContent = `${st.timestamp} · Sumber: ${st.sumber} · Metode: ${st.metode}`
      + (st.p_berisik !== null ? ` · P=${fmt(st.p_berisik, 3)}` : '');
    heroKey.textContent = st.keyakinan || (st.p_berisik !== null ? (st.p_berisik >= 0.7 ? 'SANGAT YAKIN' : 'CENDERUNG') : 'STANDAR');

    // VU Meter (Level Bar)
    const vuBar = document.getElementById('vu-bar');
    const vuText = document.getElementById('vu-val-text');
    if (st.rms_dbfs !== null && st.rms_dbfs !== undefined) {
      const db = parseFloat(st.rms_dbfs);
      vuText.textContent = `${db.toFixed(1)} dBFS`;
      // Map -60 dBFS -> 0%, 0 dBFS -> 100%
      const pct = Math.max(0, Math.min(100, ((db + 60) / 60) * 100));
      vuBar.style.width = pct + '%';
      vuBar.style.background = db >= -20 ? 'var(--merah)' : db >= -30 ? 'var(--kuning)' : 'var(--hijau)';
    }

    if (isBerisik && lastVonis !== 'BERISIK') {
      bunyikanBuzzer();
    }
    lastVonis = st.vonis;
  }

  if (stat) {
    setText('m-total', stat.total ?? '—');
    setText('m-berisik', stat.berisik ?? '—');
    setText('m-persen', stat.persen_berisik !== undefined ? stat.persen_berisik + '%' : '—');
    setText('m-tenang', stat.tenang ?? '—');
    setText('m-hari-ini', stat.hari_ini ?? '—');
    setText('m-berisik-hari-ini', stat.berisik_hari_ini !== undefined ? `${stat.berisik_hari_ini} berisik` : '—');
    setText('m-sesi', stat.sesi_aktif ?? '—');

    // Metrik Sumber Audio
    setText('m-src-realtime', stat.total_realtime ?? (stat.sumber ? stat.sumber.realtime : '0'));
    setText('m-src-rekam', stat.total_rekam ?? (stat.sumber ? stat.sumber.rekam : '0'));
    setText('m-src-upload', stat.total_upload ?? (stat.sumber ? stat.sumber.upload : '0'));
  }

  setText('txt-last-sync', d.server_time ? d.server_time.substring(11, 19) : '');
}

// ── Fetch Grafik ───────────────────────────────────────────────────────────
async function muatGrafik() {
  try {
    const res = await fetch(apiUrl('/api/grafik?menit=10'));
    if (!res.ok) return;
    const d = await res.json();
    const rows = d.data || [];
    const labels = rows.map(r => r.timestamp.substring(11, 19));
    const vals = rows.map(r => r.rms_dbfs !== null ? parseFloat(r.rms_dbfs).toFixed(1) : null);
    const refLine = rows.map(() => -20);

    chartLiveRms.data.labels = labels;
    chartLiveRms.data.datasets[0].data = vals;
    chartLiveRms.data.datasets[1].data = refLine;
    chartLiveRms.data.datasets[0].pointBackgroundColor = rows.map(
      r => r.vonis === 'BERISIK' ? COLOR_RED : COLOR_GREEN
    );
    chartLiveRms.update();
  } catch(e) {}
}

// ── Tab 2: Analisis ────────────────────────────────────────────────────────
async function muatAnalisis() {
  try {
    const res = await fetch(apiUrl('/api/analisis'));
    if (!res.ok) return;
    const json = await res.json();
    const d = json.data || {};

    const eks = d.ekstrem || {};
    setText('an-max-rms', eks.max_rms !== undefined ? `${eks.max_rms} dBFS` : '—');
    setText('an-avg-rms', eks.avg_rms !== undefined ? `${eks.avg_rms} dBFS` : '—');
    setText('an-min-rms', eks.min_rms !== undefined ? `${eks.min_rms} dBFS` : '—');

    // Chart per jam
    const jamData = d.per_jam || [];
    initChartHourly(
      jamData.map(r => r.jam + ':00'),
      jamData.map(r => r.berisik),
      jamData.map(r => r.total)
    );

    // Chart per metode
    const metData = d.per_metode || [];
    initChartMetode(
      metData.map(r => r.metode),
      metData.map(r => r.berisik),
      metData.map(r => r.tenang)
    );

    // Chart keyakinan
    const keyData = d.keyakinan || [];
    initChartKeyakinan(
      keyData.map(r => r.keyakinan),
      keyData.map(r => r.total)
    );

    // Chart sumber & Tabel sumber
    const sumData = d.per_sumber || [];
    initChartSumber(
      sumData.map(r => r.sumber),
      sumData.map(r => r.berisik),
      sumData.map(r => r.tenang !== undefined ? r.tenang : (r.total - r.berisik))
    );

    const tbodySum = document.getElementById('tbody-analisis-sumber');
    if (tbodySum) {
      if (sumData.length === 0) {
        tbodySum.innerHTML = '<tr><td colspan="6" style="text-align:center;padding:16px;color:var(--text-sub)">Belum ada data sumber audio</td></tr>';
      } else {
        tbodySum.innerHTML = sumData.map(r => {
          const tenang = r.tenang !== undefined ? r.tenang : (r.total - r.berisik);
          const rasio = r.total ? ((r.berisik / r.total) * 100).toFixed(1) : 0;
          return `
            <tr>
              <td>${formatSumberBadge(r.sumber)}</td>
              <td><b>${r.total}</b></td>
              <td style="color:var(--merah);font-weight:700">${r.berisik}</td>
              <td style="color:var(--hijau);font-weight:700">${tenang}</td>
              <td><b>${rasio}%</b></td>
              <td>${r.avg_rms !== null && r.avg_rms !== undefined ? r.avg_rms + ' dBFS' : '—'}</td>
            </tr>
          `;
        }).join('');
      }
    }
  } catch(e) {}
}

// ── Tab 3: Riwayat & Audit Log ─────────────────────────────────────────────
let debounceTimer = null;
function debounceRiwayat() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(muatRiwayat, 300);
}

async function muatRiwayat() {
  try {
    const sumber = document.getElementById('f-sumber').value;
    const vonis = document.getElementById('f-vonis').value;
    const metode = document.getElementById('f-metode').value;
    const limit = document.getElementById('f-limit').value;
    const q = document.getElementById('f-query').value;

    const url = `/api/riwayat?limit=${limit}&sumber=${encodeURIComponent(sumber)}&vonis=${encodeURIComponent(vonis)}&metode=${encodeURIComponent(metode)}&query=${encodeURIComponent(q)}`;
    const res = await fetch(apiUrl(url));
    if (!res.ok) return;
    const json = await res.json();
    const rows = json.data || [];
    const tbody = document.getElementById('tbody-riwayat');

    if (rows.length === 0) {
      tbody.innerHTML = '<tr><td colspan="10" style="text-align:center;padding:24px;color:var(--text-sub)">Tidak ada data yang sesuai filter</td></tr>';
      return;
    }

    tbody.innerHTML = rows.map(r => `
      <tr>
        <td style="color:#64748b;font-weight:700">${r.id}</td>
        <td>${r.timestamp}</td>
        <td>${formatSumberBadge(r.sumber)}</td>
        <td><b>${r.metode || '—'}</b></td>
        <td><span class="badge ${r.vonis === 'BERISIK' ? 'badge-berisik' : 'badge-tenang'}">${r.vonis}</span></td>
        <td>${r.p_berisik !== null && r.p_berisik !== undefined ? parseFloat(r.p_berisik).toFixed(3) : '—'}</td>
        <td><b>${r.rms_dbfs !== null && r.rms_dbfs !== undefined ? parseFloat(r.rms_dbfs).toFixed(1) + ' dBFS' : '—'}</b></td>
        <td style="color:var(--text-sub)">${r.keyakinan || '—'}</td>
        <td style="color:#94a3b8;font-size:11px">${r.nama_file || '—'}</td>
        <td style="font-family:monospace;font-size:11px;color:#64748b">${r.sesi_id ? r.sesi_id.substring(0, 8) + '…' : '—'}</td>
      </tr>
    `).join('');
  } catch(e) {}
}

// ── Tab 4: Sesi Pengguna ───────────────────────────────────────────────────
async function muatSesi() {
  try {
    const res = await fetch(apiUrl('/api/sesi'));
    if (!res.ok) return;
    const json = await res.json();
    const rows = json.data || [];
    const tbody = document.getElementById('tbody-sesi');

    if (rows.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;padding:24px;color:var(--text-sub)">Belum ada sesi tercatat</td></tr>';
      return;
    }

    tbody.innerHTML = rows.map(r => `
      <tr>
        <td>
          <span class="badge ${r.status === 'Online' ? 'badge-online' : r.status === 'Idle' ? 'badge-idle' : 'badge-offline'}">
            ● ${r.status}
          </span>
        </td>
        <td style="font-family:monospace;font-weight:600">${r.sesi_id}</td>
        <td>${r.mulai}</td>
        <td>${r.terakhir}</td>
        <td style="color:var(--text-sub)">${r.detik_lalu !== undefined ? `${r.detik_lalu} detik yang lalu` : '—'}</td>
      </tr>
    `).join('');
  } catch(e) {}
}

async function bersihkanSesiMati() {
  try {
    const res = await fetch(apiUrl('/api/sesi/bersihkan'), { method: 'POST' });
    if (res.ok) muatSesi();
  } catch(e) {}
}

// ── Tab 5: Sistem ──────────────────────────────────────────────────────────
async function muatSistem() {
  try {
    const res = await fetch(apiUrl('/api/sistem'));
    if (!res.ok) return;
    const json = await res.json();
    const d = json.data || {};
    setText('sys-db-path', d.db_path || '—');
    setText('sys-db-size', d.db_size_kb !== undefined ? `${d.db_size_kb}` : '—');
    setText('sys-total-rows', d.total_deteksi !== undefined ? `${d.total_deteksi}` : '—');
  } catch(e) {}
}

// ── WebSocket Realtime Engine (Auto Reconnect & Fallback) ──────────────────
let ws = null;
function sambungkanWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/live` + (adminToken ? `?token=${encodeURIComponent(adminToken)}` : '');
  
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    setText('lbl-koneksi-sidebar', 'WebSocket Realtime');
    setText('badge-sync', 'WS 1s');
  };

  ws.onmessage = (event) => {
    try {
      const d = JSON.parse(event.data);
      applyLivePayload(d);
    } catch(e) {}
  };

  ws.onclose = () => {
    setText('lbl-koneksi-sidebar', 'Polling HTTP (Fallback)');
    setText('badge-sync', 'HTTP 1s');
    // Coba sambung ulang setelah 3 detik
    setTimeout(sambungkanWebSocket, 3000);
  };

  ws.onerror = () => {
    try { ws.close(); } catch(e) {}
  };
}

// ── Polling Fallback & Periodic Refresh ─────────────────────────────────────
async function pollingStatus() {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    try {
      const res = await fetch(apiUrl('/api/status'));
      if (res.ok) {
        const d = await res.json();
        applyLivePayload(d);
      }
    } catch(e) {}
  }
}

// Setup CSV link
const btnCsv = document.getElementById('btn-csv-export');
if (btnCsv) btnCsv.href = apiUrl('/api/ekspor/csv');

// Inisialisasi awal
sambungkanWebSocket();
setInterval(pollingStatus, 1000);
setInterval(muatGrafik, 2000);
muatGrafik();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, dependencies=[Depends(verifikasi_admin)])
def dashboard():
    """Tampilkan antarmuka dashboard admin multi-menu realtime."""
    return HTMLResponse(content=DASHBOARD_HTML)
