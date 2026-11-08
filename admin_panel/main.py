"""
admin_panel/main.py
Backend FastAPI untuk dashboard admin monitoring deteksi kebisingan.
Jalankan: uvicorn admin_panel.main:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets

# Tambahkan root project ke sys.path agar bisa import deteksi_logger
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from deteksi_logger import (
    ambil_riwayat, ambil_status_terkini, ambil_data_grafik,
    hitung_statistik, inisialisasi_db
)

# ── Kredensial dari ENV (default: admin / admin123) ──────────────────────────
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "admin123")

# ── App & Security ────────────────────────────────────────────────────────────
app = FastAPI(title="Admin Panel Deteksi Kebisingan", docs_url=None, redoc_url=None)
security = HTTPBasic()

inisialisasi_db()


def verifikasi_admin(credentials: HTTPBasicCredentials = Depends(security)):
    ok_user = secrets.compare_digest(credentials.username.encode(), ADMIN_USER.encode())
    ok_pass = secrets.compare_digest(credentials.password.encode(), ADMIN_PASS.encode())
    if not (ok_user and ok_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Kredensial salah",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# ── API Endpoints ─────────────────────────────────────────────────────────────
@app.get("/api/status", dependencies=[Depends(verifikasi_admin)])
def api_status():
    """Status deteksi terkini + statistik ringkasan."""
    return JSONResponse({
        "terkini": ambil_status_terkini(),
        "statistik": hitung_statistik(),
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


@app.get("/api/riwayat", dependencies=[Depends(verifikasi_admin)])
def api_riwayat(limit: int = 50):
    """Riwayat deteksi terbaru."""
    return JSONResponse({"data": ambil_riwayat(limit=limit)})


@app.get("/api/grafik", dependencies=[Depends(verifikasi_admin)])
def api_grafik(menit: int = 10):
    """Data grafik level suara N menit terakhir."""
    return JSONResponse({"data": ambil_data_grafik(menit=menit)})


# ── Dashboard HTML ────────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Admin — Deteksi Kebisingan</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0f172a; --panel: #1e293b; --border: #334155;
    --teks: #f1f5f9; --redup: #94a3b8; --merah: #ef4444;
    --hijau: #10b981; --kuning: #f59e0b; --biru: #3b82f6;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--teks); font-family: 'Segoe UI', Arial, sans-serif; min-height: 100vh; }
  
  /* Header */
  .header { background: var(--panel); border-bottom: 1px solid var(--border); padding: 16px 28px;
            display: flex; align-items: center; justify-content: space-between; }
  .header h1 { font-size: 20px; font-weight: 700; }
  .header .badge { font-size: 11px; background: var(--biru); color: white; padding: 3px 10px;
                   border-radius: 999px; margin-left: 10px; }
  .header .update-info { font-size: 12px; color: var(--redup); }
  
  /* Layout */
  .container { max-width: 1400px; margin: 0 auto; padding: 24px 20px; }
  
  /* Status Utama */
  .status-card { border-radius: 16px; padding: 28px 32px; margin-bottom: 24px;
                 border: 1px solid var(--border); border-left-width: 8px;
                 display: flex; align-items: center; justify-content: space-between; gap: 20px;
                 transition: border-color .4s, background .4s; }
  .status-card.berisik { border-left-color: var(--merah); background: rgba(239,68,68,.07); }
  .status-card.tenang  { border-left-color: var(--hijau); background: rgba(16,185,129,.07); }
  .status-card.kosong  { border-left-color: var(--border); background: var(--panel); }
  .status-label { font-size: 11px; letter-spacing: 2px; color: var(--redup); margin-bottom: 6px; }
  .status-vonis { font-size: 42px; font-weight: 900; }
  .status-vonis.berisik { color: var(--merah); }
  .status-vonis.tenang  { color: var(--hijau); }
  .status-meta { font-size: 13px; color: var(--redup); margin-top: 6px; }
  .status-pulse { width: 18px; height: 18px; border-radius: 50%; flex-shrink: 0; }
  .status-pulse.berisik { background: var(--merah); box-shadow: 0 0 0 0 rgba(239,68,68,.5);
                           animation: pulse-red 1.2s infinite; }
  .status-pulse.tenang  { background: var(--hijau); }
  @keyframes pulse-red {
    0%  { box-shadow: 0 0 0 0 rgba(239,68,68,.5); }
    70% { box-shadow: 0 0 0 12px rgba(239,68,68,0); }
    100%{ box-shadow: 0 0 0 0 rgba(239,68,68,0); }
  }

  /* Metrik */
  .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 14px; margin-bottom: 24px; }
  .metric { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px; }
  .metric-label { font-size: 11px; color: var(--redup); margin-bottom: 6px; }
  .metric-value { font-size: 32px; font-weight: 800; }
  .metric-sub   { font-size: 11px; color: var(--redup); margin-top: 4px; }
  .m-red  { color: var(--merah); }
  .m-green{ color: var(--hijau); }
  .m-blue { color: var(--biru);  }
  .m-yellow{ color: var(--kuning);}

  /* Grid 2 kolom */
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }
  @media(max-width: 900px){ .grid2 { grid-template-columns: 1fr; } }
  
  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 20px; }
  .panel h3 { font-size: 13px; font-weight: 700; color: var(--redup); letter-spacing: 1px;
              margin-bottom: 16px; text-transform: uppercase; }
  .chart-wrap { position: relative; height: 220px; }

  /* Tabel riwayat */
  .table-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--border);
       color: var(--redup); font-size: 11px; letter-spacing: .5px; font-weight: 600; }
  td { padding: 10px 12px; border-bottom: 1px solid rgba(51,65,85,.5); vertical-align: middle; }
  tr:last-child td { border-bottom: none; }
  .badge-berisik { background: rgba(239,68,68,.2); color: var(--merah); padding: 2px 8px;
                   border-radius: 999px; font-size: 11px; font-weight: 700; }
  .badge-tenang  { background: rgba(16,185,129,.2); color: var(--hijau); padding: 2px 8px;
                   border-radius: 999px; font-size: 11px; font-weight: 700; }

  /* Loading spinner */
  .spinner { width: 32px; height: 32px; border: 3px solid var(--border);
             border-top-color: var(--biru); border-radius: 50%; animation: spin .8s linear infinite;
             margin: 40px auto; }
  @keyframes spin { to { transform: rotate(360deg); } }
  
  /* Footer */
  footer { text-align: center; color: var(--redup); font-size: 12px; padding: 20px; }
</style>
</head>
<body>

<div class="header">
  <div style="display:flex;align-items:center;gap:8px">
    <h1>🔊 Admin Panel</h1>
    <span class="badge">Deteksi Kebisingan</span>
  </div>
  <div class="update-info" id="last-update">Memuat…</div>
</div>

<div class="container">

  <!-- Status Realtime -->
  <div class="status-card kosong" id="status-card">
    <div>
      <div class="status-label">STATUS TERKINI</div>
      <div class="status-vonis" id="status-vonis">—</div>
      <div class="status-meta" id="status-meta">Belum ada data</div>
    </div>
    <div class="status-pulse" id="status-pulse" style="background:#334155"></div>
  </div>

  <!-- Metrik Statistik -->
  <div class="metrics">
    <div class="metric">
      <div class="metric-label">TOTAL DETEKSI</div>
      <div class="metric-value m-blue" id="stat-total">—</div>
      <div class="metric-sub">semua waktu</div>
    </div>
    <div class="metric">
      <div class="metric-label">BERISIK</div>
      <div class="metric-value m-red" id="stat-berisik">—</div>
      <div class="metric-sub" id="stat-persen">—</div>
    </div>
    <div class="metric">
      <div class="metric-label">TENANG</div>
      <div class="metric-value m-green" id="stat-tenang">—</div>
    </div>
    <div class="metric">
      <div class="metric-label">DETEKSI HARI INI</div>
      <div class="metric-value m-yellow" id="stat-hari-ini">—</div>
      <div class="metric-sub" id="stat-berisik-hari-ini">—</div>
    </div>
    <div class="metric">
      <div class="metric-label">SESI AKTIF</div>
      <div class="metric-value m-blue" id="stat-sesi">—</div>
      <div class="metric-sub">pengguna Streamlit</div>
    </div>
  </div>

  <!-- Grafik + Distribusi -->
  <div class="grid2">
    <div class="panel">
      <h3>📈 Level Suara (dBFS) — 10 Menit Terakhir</h3>
      <div class="chart-wrap">
        <canvas id="chart-rms"></canvas>
      </div>
    </div>
    <div class="panel">
      <h3>🥧 Distribusi Vonis</h3>
      <div class="chart-wrap">
        <canvas id="chart-pie"></canvas>
      </div>
    </div>
  </div>

  <!-- Tabel Riwayat -->
  <div class="panel">
    <h3>📋 Riwayat Deteksi Terbaru</h3>
    <div class="table-wrap">
      <div class="spinner" id="table-spinner"></div>
      <table id="tabel-riwayat" style="display:none">
        <thead>
          <tr>
            <th>#</th><th>Waktu</th><th>Sumber</th><th>Metode</th>
            <th>Vonis</th><th>P(berisik)</th><th>RMS (dBFS)</th><th>File</th>
          </tr>
        </thead>
        <tbody id="tbody-riwayat"></tbody>
      </table>
    </div>
  </div>

</div>

<footer>Refresh otomatis setiap 3 detik · Admin Panel Deteksi Kebisingan Ruangan</footer>

<script>
// ── Chart instances ────────────────────────────────────────────────────────
const MERAH  = 'rgba(239,68,68,';
const HIJAU  = 'rgba(16,185,129,';
const BIRU   = 'rgba(59,130,246,';

const ctxRms = document.getElementById('chart-rms').getContext('2d');
const chartRms = new Chart(ctxRms, {
  type: 'line',
  data: { labels: [], datasets: [{
    label: 'RMS dBFS', data: [],
    borderColor: BIRU+'1)', backgroundColor: BIRU+'.15)',
    tension: .35, fill: true, pointRadius: 3, pointHoverRadius: 5,
    borderWidth: 2,
  },{
    label: 'Berisik (-20 dBFS ref)', data: [],
    borderColor: MERAH+'0.6)', borderDash: [6,4],
    pointRadius: 0, fill: false, borderWidth: 1.5,
  }]},
  options: {
    responsive: true, maintainAspectRatio: false,
    animation: { duration: 400 },
    scales: {
      x: { ticks: { color: '#94a3b8', maxTicksLimit: 8, maxRotation: 0 },
           grid: { color: '#1e293b' }},
      y: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' },
           title: { display: true, text: 'dBFS', color: '#94a3b8' }}
    },
    plugins: { legend: { labels: { color: '#f1f5f9', font: { size: 11 } } } }
  }
});

const ctxPie = document.getElementById('chart-pie').getContext('2d');
const chartPie = new Chart(ctxPie, {
  type: 'doughnut',
  data: {
    labels: ['Berisik', 'Tenang'],
    datasets: [{ data: [0, 0],
      backgroundColor: [MERAH+'0.75)', HIJAU+'0.75)'],
      borderColor: [MERAH+'1)', HIJAU+'1)'],
      borderWidth: 2, hoverOffset: 6,
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    cutout: '65%',
    plugins: {
      legend: { position: 'bottom', labels: { color: '#f1f5f9', font: { size: 12 } } }
    },
    animation: { duration: 600 }
  }
});

// ── Helper ─────────────────────────────────────────────────────────────────
function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}
function fmt(val, dec=1) {
  if (val === null || val === undefined) return '—';
  return parseFloat(val).toFixed(dec);
}
function waktuPendek(ts) {
  if (!ts) return '—';
  return ts.substring(11, 19); // HH:MM:SS
}

// ── Fetch & Update ─────────────────────────────────────────────────────────
async function fetchJSON(url) {
  const r = await fetch(url, { headers: { 'Cache-Control': 'no-cache' } });
  if (!r.ok) throw new Error(r.status);
  return r.json();
}

async function updateStatus() {
  try {
    const d = await fetchJSON('/api/status');
    const st = d.terkini;
    const stat = d.statistik;

    // Status card
    const card   = document.getElementById('status-card');
    const vonis  = document.getElementById('status-vonis');
    const meta   = document.getElementById('status-meta');
    const pulse  = document.getElementById('status-pulse');

    if (st) {
      const berisik = st.vonis === 'BERISIK';
      card.className  = 'status-card ' + (berisik ? 'berisik' : 'tenang');
      vonis.className = 'status-vonis ' + (berisik ? 'berisik' : 'tenang');
      vonis.textContent = st.vonis;
      pulse.className = 'status-pulse ' + (berisik ? 'berisik' : 'tenang');
      meta.textContent =
        `${st.timestamp} · ${st.sumber} · ${st.metode}`
        + (st.p_berisik !== null ? ` · P=${fmt(st.p_berisik, 3)}` : '')
        + (st.rms_dbfs  !== null ? ` · ${fmt(st.rms_dbfs)} dBFS` : '');
    }

    // Statistik
    if (stat) {
      setText('stat-total',    stat.total ?? '—');
      setText('stat-berisik',  stat.berisik ?? '—');
      setText('stat-persen',   stat.persen_berisik !== undefined ? stat.persen_berisik + '% dari total' : '');
      setText('stat-tenang',   stat.tenang ?? '—');
      setText('stat-hari-ini', stat.hari_ini ?? '—');
      setText('stat-berisik-hari-ini', stat.berisik_hari_ini !== undefined ? stat.berisik_hari_ini + ' berisik hari ini' : '');
      setText('stat-sesi',     stat.sesi_aktif ?? '—');

      // Pie chart
      chartPie.data.datasets[0].data = [stat.berisik ?? 0, stat.tenang ?? 0];
      chartPie.update();
    }

    setText('last-update', 'Diperbarui: ' + d.server_time);
  } catch(e) {
    setText('last-update', 'Gagal memuat status: ' + e.message);
  }
}

async function updateGrafik() {
  try {
    const d = await fetchJSON('/api/grafik?menit=10');
    const rows = d.data || [];
    const labels = rows.map(r => waktuPendek(r.timestamp));
    const values = rows.map(r => r.rms_dbfs !== null ? parseFloat(r.rms_dbfs).toFixed(2) : null);
    const refLine = rows.map(() => -20); // garis referensi -20 dBFS

    chartRms.data.labels = labels;
    chartRms.data.datasets[0].data = values;
    chartRms.data.datasets[1].data = refLine;

    // Warnai titik berdasarkan vonis
    chartRms.data.datasets[0].pointBackgroundColor = rows.map(
      r => r.vonis === 'BERISIK' ? '#ef4444' : '#10b981'
    );
    chartRms.update();
  } catch(e) {}
}

async function updateRiwayat() {
  try {
    const d = await fetchJSON('/api/riwayat?limit=30');
    const rows = d.data || [];
    const spinner = document.getElementById('table-spinner');
    const tabel   = document.getElementById('tabel-riwayat');
    const tbody   = document.getElementById('tbody-riwayat');

    spinner.style.display = 'none';
    tabel.style.display   = 'table';
    tbody.innerHTML = rows.map((r, i) => `
      <tr>
        <td style="color:#64748b">${r.id}</td>
        <td>${r.timestamp}</td>
        <td>${r.sumber || '—'}</td>
        <td>${r.metode || '—'}</td>
        <td><span class="${r.vonis === 'BERISIK' ? 'badge-berisik' : 'badge-tenang'}">${r.vonis}</span></td>
        <td>${r.p_berisik !== null && r.p_berisik !== undefined ? parseFloat(r.p_berisik).toFixed(3) : '—'}</td>
        <td>${r.rms_dbfs !== null && r.rms_dbfs !== undefined ? parseFloat(r.rms_dbfs).toFixed(1) : '—'}</td>
        <td style="color:#94a3b8;font-size:12px">${r.nama_file || '—'}</td>
      </tr>
    `).join('');
  } catch(e) {}
}

// ── Loop refresh ──────────────────────────────────────────────────────────
async function refresh() {
  await Promise.all([updateStatus(), updateGrafik(), updateRiwayat()]);
}

refresh();
setInterval(refresh, 3000); // refresh setiap 3 detik
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, dependencies=[Depends(verifikasi_admin)])
def dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)
