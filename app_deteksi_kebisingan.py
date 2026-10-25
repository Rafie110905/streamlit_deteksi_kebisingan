import hashlib
import os
import threading
from collections import deque
import numpy as np
import pandas as pd
import streamlit as st
import librosa
import librosa.display
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Deteksi Kebisingan — Random Forest", page_icon="🌲", layout="wide")

FRAME_LEN, HOP_LEN, SEED = 2048, 512, 42
FITUR_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fitur_audio_kebisingan.csv")
WARNA = {"berisik": "#f43f5e", "tenang": "#10b981", "ragu": "#f59e0b",
         "gelap": "#0f172a", "panel": "#1e293b", "teks": "#e2e8f0", "redup": "#94a3b8"}


# ------------------------------------------------------------------ model
@st.cache_resource(show_spinner="Melatih Random Forest dari data fitur…")
def siapkan_model():
    df = pd.read_csv(FITUR_CSV)
    kolom = [c for c in df.columns if c not in ("file_name", "label", "grup", "path")]
    X = df[kolom].values
    y = (df["label"] == "Berisik").astype(int).values
    grup = df["grup"].astype(str).values

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    rf_cv = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1)
    akurasi_cv = accuracy_score(y, cross_val_predict(rf_cv, X, y, cv=cv, groups=grup))

    model = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=SEED, n_jobs=-1).fit(X, y)
    imp = pd.Series(model.feature_importances_, index=kolom).sort_values(ascending=False)
    ref = {
        "mu": df[kolom].mean(), "sd": df[kolom].std().replace(0, 1),
        "mu_b": df[df.label == "Berisik"][kolom].mean(),
        "mu_t": df[df.label == "Tidak Berisik"][kolom].mean(),
    }
    return model, kolom, imp, ref, akurasi_cv, len(df)


# ------------------------------------------------------------------ audio
def muat_audio(path):
    try:
        y, sr = sf.read(path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
    except Exception:
        y, sr = librosa.load(path, sr=None, mono=True)
    return np.ascontiguousarray(y, dtype=np.float32), sr


def ekstrak_fitur(path):
    y, sr = muat_audio(path)
    return ekstrak_fitur_sinyal(y, sr)


def ekstrak_fitur_sinyal(y, sr):
    y = np.ascontiguousarray(y, dtype=np.float32)
    kw = dict(n_fft=FRAME_LEN, hop_length=HOP_LEN)
    d = {
        "rms": librosa.feature.rms(y=y, frame_length=FRAME_LEN, hop_length=HOP_LEN)[0],
        "zcr": librosa.feature.zero_crossing_rate(y=y, frame_length=FRAME_LEN, hop_length=HOP_LEN)[0],
        "centroid": librosa.feature.spectral_centroid(y=y, sr=sr, **kw)[0],
        "rolloff": librosa.feature.spectral_rolloff(y=y, sr=sr, **kw)[0],
        "bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=sr, **kw)[0],
        "flatness": librosa.feature.spectral_flatness(y=y, **kw)[0],
    }
    f = {}
    for nama, v in d.items():
        f[f"{nama}_mean"] = float(np.mean(v)); f[f"{nama}_std"] = float(np.std(v))
        f[f"{nama}_median"] = float(np.median(v)); f[f"{nama}_max"] = float(np.max(v))
    f["rms_dbfs"] = float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12))
    return f, y, sr, d


class PenyanggaAudioLangsung:
    """Menampung frame audio dari browser (WebRTC) dan memprediksi tiap 1 detik
    dari jendela 2 detik terakhir. Prediksi dijalankan di thread terpisah supaya
    callback audio tidak tersendat."""
    JENDELA_DETIK = 2

    def __init__(self):
        self._lock = threading.Lock()
        self._reset_buffer()
        self.hasil = None
        self.error = None
        self.nomor_hasil = 0
        self.jumlah_frame = 0
        self.level_dbfs = None
        self.sr_terakhir = None

    def _reset_buffer(self):
        self._potongan = deque()
        self._jumlah_sampel = 0
        self._sampel_sejak_prediksi = 0
        self._sr = None
        self._sedang_hitung = False

    def reset(self):
        with self._lock:
            self._reset_buffer()
            self.hasil = None
            self.error = None
            self.nomor_hasil = 0
            self.jumlah_frame = 0
            self.level_dbfs = None
            self.sr_terakhir = None

    @staticmethod
    def _ke_mono(frame):
        """Ubah frame WebRTC -> array mono float32 [-1, 1].
        PENTING: frame dari browser biasanya 's16' stereo *packed* (selang-seling L,R),
        sehingga to_ndarray() berbentuk (1, 2*jumlah_sampel). Harus di-reshape per kanal."""
        arr = frame.to_ndarray()
        n_ch = max(1, len(frame.layout.channels))
        if arr.dtype.kind in "iu":
            batas = max(abs(np.iinfo(arr.dtype).min), np.iinfo(arr.dtype).max)
            arr = arr.astype(np.float32) / batas
        else:
            arr = arr.astype(np.float32, copy=False)
        if frame.format.is_planar:          # bentuk (kanal, sampel)
            mono = arr.mean(axis=0) if arr.ndim > 1 else arr
        else:                               # packed: (1, sampel * kanal)
            mono = arr.reshape(-1, n_ch).mean(axis=1)
        return np.nan_to_num(mono, copy=False)

    def proses_frame(self, frame, model, kolom):
        try:
            data = self._ke_mono(frame)
            sr = frame.sample_rate
            if not sr or data.size == 0:
                return
            level = float(20 * np.log10(np.sqrt(np.mean(data ** 2)) + 1e-12))

            with self._lock:
                if sr != self._sr:
                    self._reset_buffer()
                    self._sr = sr
                self.sr_terakhir = sr
                self.level_dbfs = level
                self.jumlah_frame += 1
                self._potongan.append(data.copy())
                self._jumlah_sampel += data.size
                self._sampel_sejak_prediksi += data.size

                ukuran_jendela = int(sr * self.JENDELA_DETIK)
                while self._jumlah_sampel > ukuran_jendela:
                    lebih = self._jumlah_sampel - ukuran_jendela
                    awal = self._potongan[0]
                    if lebih >= awal.size:
                        self._potongan.popleft()
                        self._jumlah_sampel -= awal.size
                    else:
                        self._potongan[0] = awal[lebih:]
                        self._jumlah_sampel -= lebih

                if (self._jumlah_sampel < ukuran_jendela
                        or self._sampel_sejak_prediksi < sr
                        or self._sedang_hitung):
                    return
                y = np.concatenate(tuple(self._potongan))
                self._sampel_sejak_prediksi = 0
                self._sedang_hitung = True

            # hitung di thread lain -> callback audio langsung kembali
            threading.Thread(target=self._hitung, args=(y, sr, model, kolom), daemon=True).start()
        except Exception as e:
            with self._lock:
                self.error = str(e)

    def _hitung(self, y, sr, model, kolom):
        try:
            f, _, _, d = ekstrak_fitur_sinyal(y, sr)
            x = pd.DataFrame([f])[kolom].values.astype(np.float32)
            idx1 = list(model.classes_).index(1)
            p = float(model.predict_proba(x)[0][idx1])
            suara = np.array([pohon.predict_proba(x)[0][idx1] for pohon in model.estimators_])
            hasil = {
                "fitur": f, "p": p, "detail": d, "sr": sr, "durasi": len(y) / sr,
                "n_berisik": int((suara >= .5).sum()), "n_pohon": len(suara),
            }
            with self._lock:
                self.hasil = hasil
                self.error = None
                self.nomor_hasil += 1
        except Exception as e:
            with self._lock:
                self.error = str(e)
        finally:
            with self._lock:
                self._sedang_hitung = False

    def baca(self):
        with self._lock:
            return self.hasil, self.error, self.nomor_hasil, self.jumlah_frame

    def info_mic(self):
        with self._lock:
            return self.level_dbfs, self.sr_terakhir


def tingkat_keyakinan(p):
    j = abs(p - 0.5)
    return "SANGAT YAKIN" if j >= .35 else "YAKIN" if j >= .20 else "CENDERUNG" if j >= .10 else "RAGU-RAGU"


def tentukan_vonis(p, rms_dbfs, ambang_rms_dbfs):
    override_rms = p < 0.5 and rms_dbfs >= ambang_rms_dbfs
    return p >= 0.5 or override_rms, override_rms


def kartu_hasil(nama, p, n_berisik, n_pohon, f, berisik, override_rms, ambang_rms_dbfs):
    ragu = abs(p - .5) < .10
    warna = WARNA["ragu"] if ragu else WARNA["berisik"] if berisik else WARNA["tenang"]
    teks, ikon = ("BERISIK", "🚨") if berisik else ("TIDAK BERISIK", "🌿")
    pesan_override = (
        f'<div style="font-size:12px;color:{WARNA["berisik"]};margin-top:6px">'
        f"Diputuskan BERISIK karena RMS ≥ {ambang_rms_dbfs:.1f} dBFS "
        f"(P model = {p:.3f})</div>"
        if override_rms else ""
    )
    kotak = lambda k, v, s="": (f"<div style='background:{WARNA['panel']};padding:8px 10px;border-radius:8px'>"
                                 f"<div style='color:{WARNA['redup']};font-size:10px'>{k}</div><b>{v}</b>"
                                 f"<div style='color:{WARNA['redup']};font-size:10px'>{s}</div></div>")
    return f"""
    <div style="background:{WARNA['gelap']};color:{WARNA['teks']};padding:20px 24px;border-radius:14px;
                border-left:8px solid {warna};border:1px solid #334155;border-left:8px solid {warna};font-family:'Segoe UI',Arial,sans-serif">
      <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:12px">
        <div>
          <div style="font-size:11px;letter-spacing:2px;color:{WARNA['redup']}">🌲 RANDOM FOREST · {nama}</div>
          <div style="font-size:32px;font-weight:800;color:{warna};margin-top:4px">{ikon} {teks}</div>
          <div style="font-size:13px;color:{WARNA['redup']}">Keyakinan model: <b style="color:{WARNA['teks']}">{tingkat_keyakinan(p)}</b></div>{pesan_override}
        </div>
        <div style="text-align:right">
          <div style="font-size:11px;color:{WARNA['redup']}">SUARA POHON (MODEL)</div>
          <div style="font-size:32px;font-weight:800">{n_berisik}<span style="font-size:15px;color:{WARNA['redup']}"> / {n_pohon}</span></div>
          <div style="font-size:11px;color:{WARNA['redup']}">memilih BERISIK</div>
        </div>
      </div>
      <div style="margin-top:16px">
        <div style="display:flex;justify-content:space-between;font-size:11px;color:{WARNA['redup']}">
          <span>🌿 Tenang</span><span>P(berisik) = <b style="color:{WARNA['teks']}">{p:.3f}</b></span><span>Berisik 🚨</span></div>
        <div style="position:relative;height:14px;border-radius:7px;margin-top:5px;
             background:linear-gradient(90deg,{WARNA['tenang']} 0%,{WARNA['tenang']} 40%,{WARNA['ragu']} 40%,{WARNA['ragu']} 60%,{WARNA['berisik']} 60%,{WARNA['berisik']} 100%)">
          <div style="position:absolute;left:{p*100:.1f}%;top:-5px;width:4px;height:24px;background:#fff;border-radius:2px;transform:translateX(-2px)"></div>
        </div>
        <div style="text-align:center;font-size:10px;color:{WARNA['redup']};margin-top:3px">zona kuning = ragu (0.40 – 0.60)</div>
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:8px;margin-top:14px;font-size:12px">
        {kotak('RMS', f"{f['rms_mean']:.4f}", f"{f['rms_dbfs']:.1f} dBFS")}
        {kotak('ZCR', f"{f['zcr_mean']:.4f}")}
        {kotak('CENTROID', f"{f['centroid_mean']:.0f} Hz")}
        {kotak('ROLLOFF', f"{f['rolloff_mean']:.0f} Hz")}
        {kotak('FLATNESS', f"{f['flatness_mean']:.5f}")}
      </div>
    </div>"""


def buat_dashboard(f, y, sr, d, p, berisik, n_berisik, n_pohon, kolom, imp, ref):
    top = list(imp.head(8).index)
    warna = WARNA["ragu"] if abs(p - .5) < .10 else WARNA["berisik"] if berisik else WARNA["tenang"]
    fig = plt.figure(figsize=(14, 8.2), facecolor=WARNA["gelap"])
    gs = fig.add_gridspec(2, 2, hspace=.42, wspace=.28)

    def gaya(ax, judul):
        ax.set_facecolor(WARNA["panel"]); ax.set_title(judul, color=WARNA["teks"], fontsize=11, loc="left", fontweight="bold")
        ax.tick_params(colors=WARNA["redup"], labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#334155")
        ax.grid(color="#334155", alpha=.5, lw=.6)

    ax1 = fig.add_subplot(gs[0, 0]); gaya(ax1, f"① Voting {n_pohon} pohon")
    jml = [n_pohon - n_berisik, n_berisik]
    bars = ax1.bar(["Tidak Berisik", "Berisik"], jml, color=[WARNA["tenang"], WARNA["berisik"]], width=.55)
    for b, j in zip(bars, jml):
        ax1.text(b.get_x() + b.get_width() / 2, j + n_pohon * .015, f"{j} ({j / n_pohon * 100:.0f}%)",
                 ha="center", color=WARNA["teks"], fontsize=11, fontweight="bold")
    ax1.axhline(n_pohon / 2, color="white", ls="--", lw=1, alpha=.7)
    ax1.set_ylim(0, n_pohon * 1.12); ax1.tick_params(axis="x", labelcolor=WARNA["teks"], labelsize=10)

    ax2 = fig.add_subplot(gs[0, 1]); gaya(ax2, "② Profil fitur penting (z-score vs dataset)")
    z_s = ((pd.Series(f)[top] - ref["mu"][top]) / ref["sd"][top]).values
    z_b = ((ref["mu_b"][top] - ref["mu"][top]) / ref["sd"][top]).values
    z_t = ((ref["mu_t"][top] - ref["mu"][top]) / ref["sd"][top]).values
    yy = np.arange(len(top))[::-1]
    ax2.barh(yy, z_s, height=.55, color=[WARNA["berisik"] if abs(a - b) < abs(a - c) else WARNA["tenang"]
                                         for a, b, c in zip(z_s, z_b, z_t)])
    ax2.scatter(z_b, yy, marker="D", s=38, color=WARNA["berisik"], edgecolor="white", zorder=3, label="rata-rata Berisik")
    ax2.scatter(z_t, yy, marker="o", s=38, color=WARNA["tenang"], edgecolor="white", zorder=3, label="rata-rata Tenang")
    ax2.set_yticks(yy); ax2.set_yticklabels(top, fontsize=8, color=WARNA["teks"]); ax2.axvline(0, color="white", lw=.8)
    ax2.legend(facecolor=WARNA["panel"], labelcolor=WARNA["teks"], fontsize=7.5, loc="lower right")

    t = librosa.times_like(d["rms"], sr=sr, hop_length=HOP_LEN)
    ax3 = fig.add_subplot(gs[1, 0]); gaya(ax3, "③ Gelombang & amplop RMS")
    ax3.plot(np.arange(len(y)) / sr, y, lw=.4, color="#64748b")
    ax3.plot(t, d["rms"], color=warna, lw=1.8); ax3.plot(t, -d["rms"], color=warna, lw=1.8)
    ax3.set_xlabel("Waktu (detik)", color=WARNA["redup"], fontsize=9)

    ax4 = fig.add_subplot(gs[1, 1]); gaya(ax4, "④ Spektrogram & Spectral Centroid")
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=FRAME_LEN, hop_length=HOP_LEN)), ref=np.max)
    librosa.display.specshow(S, sr=sr, hop_length=HOP_LEN, x_axis="time", y_axis="linear", cmap="viridis", ax=ax4)
    ax4.plot(t, d["centroid"], color="white", lw=1.4); ax4.set_ylim(0, min(sr / 2, 12000))
    ax4.set_xlabel("Waktu (detik)", color=WARNA["redup"], fontsize=9); ax4.set_ylabel("Hz", color=WARNA["redup"], fontsize=9)
    return fig


# ------------------------------------------------------------------ Metode RMS (aturan)
def analisis_rms_aturan(d, sr, durasi, rms_t=0.050, cent_t=1600.0, zcr_t=0.040, skor_t=0.70, persen_t=40.0):
    """Aturan sensor dari notebook awal: pemicu per-frame -> penghalusan -> persen durasi bising + skor rata-rata."""
    rms_f, cent_f, zcr_f = d["rms"], d["centroid"], d["zcr"]
    cond_loud = rms_f >= rms_t
    cond_harsh = (rms_f >= rms_t * 0.7) & (cent_f >= cent_t)
    cond_zcr = (rms_f >= rms_t * 0.5) & (zcr_f >= zcr_t)
    mentah = (cond_loud | cond_harsh | cond_zcr).astype(int)
    halus = (np.convolve(mentah, np.ones(5) / 5, mode="same") >= 0.5).astype(int) if len(mentah) >= 5 else mentah

    persen = float(np.sum(halus) * (HOP_LEN / sr) / (durasi + 1e-9) * 100.0)
    m_rms, m_cent, m_zcr = float(np.mean(rms_f)), float(np.mean(cent_f)), float(np.mean(zcr_f))
    skor = (m_rms / 0.08) * 0.45 + (m_cent / 2200.0) * 0.35 + (m_zcr / 0.05) * 0.20
    oleh_skor, oleh_persen = skor >= skor_t, persen >= persen_t
    return {"berisik": bool(oleh_skor or oleh_persen), "oleh_skor": bool(oleh_skor), "oleh_persen": bool(oleh_persen),
            "skor": float(skor), "persen": persen, "mentah": mentah, "halus": halus,
            "m_rms": m_rms, "m_cent": m_cent, "m_zcr": m_zcr, "skor_t": skor_t, "persen_t": persen_t}


def kartu_rms(nama, r, prm):
    berisik = r["berisik"]
    warna = WARNA["berisik"] if berisik else WARNA["tenang"]
    teks, ikon = ("BERISIK", "🚨") if berisik else ("TIDAK BERISIK", "🌿")
    alasan = []
    if r["oleh_skor"]: alasan.append(f"skor {r['skor']:.2f} ≥ {r['skor_t']:.2f}")
    if r["oleh_persen"]: alasan.append(f"durasi bising {r['persen']:.0f}% ≥ {r['persen_t']:.0f}%")
    ket = ("Terpicu: " + " · ".join(alasan)) if alasan else "Skor dan durasi bising di bawah ambang"
    skala = 1.4
    pos = min(r["skor"] / skala, 1.0) * 100
    tanda = min(r["skor_t"] / skala, 1.0) * 100
    kotak = lambda k, v, s="": (f"<div style='background:{WARNA['panel']};padding:8px 10px;border-radius:8px'>"
                                 f"<div style='color:{WARNA['redup']};font-size:10px'>{k}</div><b>{v}</b>"
                                 f"<div style='color:{WARNA['redup']};font-size:10px'>{s}</div></div>")
    return f"""
    <div style="background:{WARNA['gelap']};color:{WARNA['teks']};padding:20px 24px;border-radius:14px;
                border:1px solid #334155;border-left:8px solid {warna};font-family:'Segoe UI',Arial,sans-serif">
      <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:12px">
        <div>
          <div style="font-size:11px;letter-spacing:2px;color:{WARNA['redup']}">📏 RMS (ATURAN) · {nama}</div>
          <div style="font-size:32px;font-weight:800;color:{warna};margin-top:4px">{ikon} {teks}</div>
          <div style="font-size:13px;color:{WARNA['redup']}">{ket}</div>
        </div>
        <div style="text-align:right">
          <div style="font-size:11px;color:{WARNA['redup']}">DURASI BISING</div>
          <div style="font-size:32px;font-weight:800">{r['persen']:.0f}<span style="font-size:15px;color:{WARNA['redup']}"> %</span></div>
          <div style="font-size:11px;color:{WARNA['redup']}">ambang {r['persen_t']:.0f}%</div>
        </div>
      </div>
      <div style="margin-top:16px">
        <div style="display:flex;justify-content:space-between;font-size:11px;color:{WARNA['redup']}">
          <span>🌿 Tenang</span><span>Skor berisik = <b style="color:{WARNA['teks']}">{r['skor']:.2f}</b> (ambang {r['skor_t']:.2f})</span><span>Berisik 🚨</span></div>
        <div style="position:relative;height:14px;border-radius:7px;margin-top:5px;
             background:linear-gradient(90deg,{WARNA['tenang']} 0%,{WARNA['tenang']} {tanda:.1f}%,{WARNA['berisik']} {tanda:.1f}%,{WARNA['berisik']} 100%)">
          <div style="position:absolute;left:{pos:.1f}%;top:-5px;width:4px;height:24px;background:#fff;border-radius:2px;transform:translateX(-2px)"></div>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:8px;margin-top:14px;font-size:12px">
        {kotak('RMS rata-rata', f"{r['m_rms']:.4f}", f"ambang {prm['rms_t']:.3f}")}
        {kotak('CENTROID', f"{r['m_cent']:.0f} Hz", f"ambang {prm['cent_t']:.0f}")}
        {kotak('ZCR', f"{r['m_zcr']:.4f}", f"ambang {prm['zcr_t']:.3f}")}
      </div>
    </div>"""


def buat_dashboard_rms(y, sr, d, r, prm):
    warna = WARNA["berisik"] if r["berisik"] else WARNA["tenang"]
    t = librosa.times_like(d["rms"], sr=sr, hop_length=HOP_LEN)
    fig = plt.figure(figsize=(14, 8.2), facecolor=WARNA["gelap"])
    gs = fig.add_gridspec(2, 2, hspace=.42, wspace=.28)

    def gaya(ax, judul):
        ax.set_facecolor(WARNA["panel"]); ax.set_title(judul, color=WARNA["teks"], fontsize=11, loc="left", fontweight="bold")
        ax.tick_params(colors=WARNA["redup"], labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#334155")
        ax.grid(color="#334155", alpha=.5, lw=.6)

    def blok_bising(ax):
        ax.fill_between(t, 0, 1, where=r["halus"].astype(bool), transform=ax.get_xaxis_transform(),
                        color=WARNA["berisik"], alpha=.18, lw=0)

    ax1 = fig.add_subplot(gs[0, 0]); gaya(ax1, "① Gelombang (area merah = terdeteksi bising)")
    ax1.plot(np.arange(len(y)) / sr, y, lw=.4, color="#64748b"); blok_bising(ax1)
    ax1.set_xlabel("Waktu (detik)", color=WARNA["redup"], fontsize=9)

    ax2 = fig.add_subplot(gs[0, 1]); gaya(ax2, "② RMS per frame vs ambang")
    ax2.plot(t, d["rms"], color=warna, lw=1.5); blok_bising(ax2)
    ax2.axhline(prm["rms_t"], color=WARNA["berisik"], ls="--", lw=1.3, label=f"ambang RMS {prm['rms_t']:.3f}")
    ax2.axhline(prm["rms_t"] * .5, color=WARNA["ragu"], ls=":", lw=1, label="0,5× (jalur ZCR)")
    ax2.legend(facecolor=WARNA["panel"], labelcolor=WARNA["teks"], fontsize=8)
    ax2.set_xlabel("Waktu (detik)", color=WARNA["redup"], fontsize=9)

    ax3 = fig.add_subplot(gs[1, 0]); gaya(ax3, "③ Spectral Centroid vs ambang")
    ax3.plot(t, d["centroid"], color="#38bdf8", lw=1.3); blok_bising(ax3)
    ax3.axhline(prm["cent_t"], color=WARNA["berisik"], ls="--", lw=1.3, label=f"ambang {prm['cent_t']:.0f} Hz")
    ax3.legend(facecolor=WARNA["panel"], labelcolor=WARNA["teks"], fontsize=8)
    ax3.set_xlabel("Waktu (detik)", color=WARNA["redup"], fontsize=9); ax3.set_ylabel("Hz", color=WARNA["redup"], fontsize=9)

    ax4 = fig.add_subplot(gs[1, 1]); gaya(ax4, "④ ZCR vs ambang & pemicu")
    ax4.plot(t, d["zcr"], color="#c084fc", lw=1.3); blok_bising(ax4)
    ax4.axhline(prm["zcr_t"], color=WARNA["berisik"], ls="--", lw=1.3, label=f"ambang ZCR {prm['zcr_t']:.3f}")
    ax4.step(t, r["mentah"] * (d["zcr"].max() * .12), where="mid", color=WARNA["ragu"], lw=1, alpha=.9, label="pemicu mentah")
    ax4.legend(facecolor=WARNA["panel"], labelcolor=WARNA["teks"], fontsize=8)
    ax4.set_xlabel("Waktu (detik)", color=WARNA["redup"], fontsize=9)
    return fig


# ------------------------------------------------------------------ WebRTC
def konfigurasi_rtc():
    """Beberapa server STUN publik. Jika jaringan memblokir koneksi langsung (umum di
    jaringan kampus / data seluler), tambahkan TURN lewat st.secrets (lihat catatan)."""
    servers = [
        {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302",
                  "stun:stun2.l.google.com:19302", "stun:stun.cloudflare.com:3478"]},
    ]
    try:  # opsional: [turn] url, username, credential di .streamlit/secrets.toml
        t = st.secrets["turn"]
        servers.append({"urls": [t["url"]], "username": t["username"], "credential": t["credential"]})
    except Exception:
        pass
    return {"iceServers": servers}


# ------------------------------------------------------------------ UI
model, kolom, imp, ref, akurasi_cv, n_data = siapkan_model()

M_RF, M_RMS, M_BOTH = "🌲 Random Forest", "📏 RMS (aturan)", "⚖️ Bandingkan keduanya"

with st.sidebar:
    st.header("ℹ️ Tentang Metode")
    st.markdown("**🌲 Random Forest**")
    st.metric("Akurasi validasi silang", f"{akurasi_cv * 100:.1f}%")
    st.caption(f"300 pohon · {len(kolom)} fitur · {n_data} data latih. Validasi 5-fold dikelompokkan per mahasiswa.")
    st.markdown("**📏 RMS (aturan)**")
    st.caption("Tanpa pelatihan. Frame dianggap bising bila RMS tinggi, atau RMS sedang + centroid tinggi, "
               "atau RMS rendah-sedang + ZCR tinggi. Berisik bila skor ≥ ambang skor atau durasi bising ≥ ambang persen.")
    st.markdown("**Fitur terpenting (RF)**")
    st.bar_chart(imp.head(8)[::-1])
    st.caption("Hasil bersifat indikatif. Rekaman yang sangat berbeda dari data latih bisa salah diklasifikasi.")

st.title("🔊 Deteksi Kebisingan Ruangan")
st.write("Unggah rekaman, rekam langsung, atau deteksi realtime. Pilih metode **Random Forest**, **RMS (aturan)**, "
         "atau **bandingkan keduanya**.")

for kunci, nilai in [("riwayat", []), ("hasil_audio", None)]:
    if kunci not in st.session_state:
        st.session_state[kunci] = nilai
if "audio_langsung" not in st.session_state:
    st.session_state.audio_langsung = PenyanggaAudioLangsung()

tab_uji, tab_riw = st.tabs(["🎙️ Uji Sampel", "🗂️ Riwayat"])

with tab_uji:
    metode = st.radio("Metode deteksi", [M_RF, M_RMS, M_BOTH], horizontal=True)

    pakai_override, ambang_rms_dbfs = False, -30.0
    if metode in (M_RF, M_BOTH):
        with st.expander("⚙️ Pengaturan Random Forest"):
            pakai_override = st.checkbox(
                "Perkuat dengan ambang level suara (dBFS)", value=True,
                help="Jika model menjawab TIDAK BERISIK tetapi level rekaman ≥ ambang, hasil diubah menjadi BERISIK.")
            ambang_rms_dbfs = st.slider("Ambang suara keras (RMS, dBFS)", -40.0, -10.0, -30.0, 0.5,
                                        disabled=not pakai_override)
            st.caption("dBFS adalah level digital rekaman, bukan dB SPL; dipengaruhi mikrofon, jarak, dan gain.")

    prm = dict(rms_t=0.050, cent_t=1600.0, zcr_t=0.040, skor_t=0.70, persen_t=40.0)
    if metode in (M_RMS, M_BOTH):
        with st.expander("⚙️ Pengaturan RMS (aturan)"):
            c1, c2, c3 = st.columns(3)
            prm["rms_t"] = c1.slider("Ambang RMS", 0.010, 0.200, 0.050, 0.005, format="%.3f")
            prm["cent_t"] = c2.slider("Ambang centroid (Hz)", 800.0, 4000.0, 1600.0, 50.0)
            prm["zcr_t"] = c3.slider("Ambang ZCR", 0.010, 0.150, 0.040, 0.005, format="%.3f")
            c4, c5 = st.columns(2)
            prm["skor_t"] = c4.slider("Ambang skor berisik", 0.30, 1.20, 0.70, 0.05)
            prm["persen_t"] = c5.slider("Ambang durasi bising (%)", 10.0, 90.0, 40.0, 5.0)

    sumber_audio = st.radio("Sumber audio", ["Unggah berkas", "Rekam langsung", "Deteksi realtime"], horizontal=True)
    if sumber_audio == "Unggah berkas":
        audio = st.file_uploader("Pilih berkas audio", type=["mp3", "wav", "m4a", "flac", "ogg"])
        nama_audio = audio.name if audio is not None else ""
    elif sumber_audio == "Rekam langsung":
        audio = st.audio_input("Rekam suara menggunakan mikrofon")
        nama_audio = "Rekaman langsung.wav" if audio is not None else ""

    def tampilkan_kartu(nama, p, n_b, n_p, f, d, sr, durasi):
        """Kartu hasil sesuai metode terpilih. Mengembalikan (vonis_rf, override, rms_hasil)."""
        berisik_rf, override = tentukan_vonis(p, f["rms_dbfs"], ambang_rms_dbfs) if pakai_override else (p >= 0.5, False)
        rms_hasil = analisis_rms_aturan(d, sr, durasi, **prm)
        if metode == M_RF:
            st.markdown(kartu_hasil(nama, p, n_b, n_p, f, berisik_rf, override, ambang_rms_dbfs), unsafe_allow_html=True)
        elif metode == M_RMS:
            st.markdown(kartu_rms(nama, rms_hasil, prm), unsafe_allow_html=True)
        else:
            k1, k2 = st.columns(2)
            k1.markdown(kartu_hasil(nama, p, n_b, n_p, f, berisik_rf, override, ambang_rms_dbfs), unsafe_allow_html=True)
            k2.markdown(kartu_rms(nama, rms_hasil, prm), unsafe_allow_html=True)
            if berisik_rf == rms_hasil["berisik"]:
                st.success("✅ Kedua metode sepakat: " + ("BERISIK" if berisik_rf else "TIDAK BERISIK"))
            else:
                st.warning(f"⚠️ Kedua metode berbeda — Random Forest: {'BERISIK' if berisik_rf else 'TIDAK BERISIK'} · "
                           f"RMS: {'BERISIK' if rms_hasil['berisik'] else 'TIDAK BERISIK'}")
        return berisik_rf, override, rms_hasil

    if sumber_audio == "Deteksi realtime":
        from streamlit_webrtc import webrtc_streamer

        st.caption("Tekan START dan izinkan mikrofon. Hasil diperbarui setiap detik dari 2 detik suara terakhir.")
        st.session_state.hasil_audio = None
        state_audio = st.session_state.audio_langsung

        def callback_audio(frame):
            state_audio.proses_frame(frame, model, kolom)
            return frame

        webrtc_ctx = webrtc_streamer(
            key="deteksi-kebisingan-realtime",
            audio_frame_callback=callback_audio,
            media_stream_constraints={"audio": True, "video": False},
            rtc_configuration=konfigurasi_rtc(),
            sendback_audio=False,
        )

        @st.fragment(run_every="1s")
        def tampilkan_hasil_realtime():
            hasil_live, error_live, nomor_hasil, jumlah_frame = state_audio.baca()
            if not webrtc_ctx.state.playing:
                st.info("Tekan START untuk mulai mendeteksi melalui mikrofon. Tekan STOP untuk mengakhiri.")
                return
            if error_live:
                st.error(f"Gagal menganalisis audio langsung: {error_live}")
            if hasil_live is None:
                if jumlah_frame:
                    level, sr_live = state_audio.info_mic()
                    st.info(f"Audio diterima ({jumlah_frame} frame, {sr_live} Hz, level {level:.1f} dBFS). "
                            "Menunggu 2 detik suara untuk prediksi pertama…")
                else:
                    st.warning("Koneksi terbuka tetapi belum ada audio masuk. Periksa izin mikrofon, "
                               "perangkat input di browser, atau jaringan (butuh STUN/TURN).")
                return
            tampilkan_kartu("Deteksi realtime", hasil_live["p"], hasil_live["n_berisik"], hasil_live["n_pohon"],
                            hasil_live["fitur"], hasil_live["detail"], hasil_live["sr"], hasil_live["durasi"])
            st.caption(f"Level: {hasil_live['fitur']['rms_dbfs']:.1f} dBFS · P(RF): {hasil_live['p']:.3f} · "
                       f"Pembaruan ke-{nomor_hasil} · frame diterima: {jumlah_frame}")

        tampilkan_hasil_realtime()
        if not webrtc_ctx.state.playing:
            state_audio.reset()

    elif audio is not None:
        st.audio(audio)
        audio_bytes = audio.getvalue()
        audio_id = hashlib.sha256(audio_bytes).hexdigest()
        hasil = st.session_state.hasil_audio
        audio_baru = hasil is None or hasil["audio_id"] != audio_id

        if audio_baru:
            import tempfile
            sufiks = os.path.splitext(nama_audio)[1] or ".mp3"
            with tempfile.NamedTemporaryFile(delete=False, suffix=sufiks) as tmp:
                tmp.write(audio_bytes); jalur = tmp.name
            try:
                with st.spinner("Mengekstrak fitur & memprediksi…"):
                    f, y, sr, d = ekstrak_fitur(jalur)
                    x = pd.DataFrame([f])[kolom].values.astype(np.float32)
                    idx1 = list(model.classes_).index(1)
                    p = float(model.predict_proba(x)[0][idx1])
                    suara = np.array([t.predict_proba(x)[0][idx1] for t in model.estimators_])
                    n_b, n_p = int((suara >= .5).sum()), len(suara)
                hasil = {"audio_id": audio_id, "nama": nama_audio, "fitur": f, "y": y, "sr": sr, "detail": d,
                         "durasi": len(y) / sr, "p": p, "n_berisik": n_b, "n_pohon": n_p}
                st.session_state.hasil_audio = hasil
            except Exception as e:
                hasil = {"audio_id": audio_id, "error": str(e)}
                st.session_state.hasil_audio = hasil
            finally:
                if os.path.exists(jalur):
                    os.remove(jalur)

        hasil = st.session_state.hasil_audio
        if hasil and hasil["audio_id"] == audio_id and "error" not in hasil:
            f, y, sr, d = hasil["fitur"], hasil["y"], hasil["sr"], hasil["detail"]
            p, n_b, n_p, durasi = hasil["p"], hasil["n_berisik"], hasil["n_pohon"], hasil["durasi"]
            berisik_rf, override, rms_hasil = tampilkan_kartu(nama_audio, p, n_b, n_p, f, d, sr, durasi)
            st.write("")

            if metode == M_RF:
                fig = buat_dashboard(f, y, sr, d, p, berisik_rf, n_b, n_p, kolom, imp, ref)
                st.pyplot(fig); plt.close(fig)
            elif metode == M_RMS:
                fig = buat_dashboard_rms(y, sr, d, rms_hasil, prm)
                st.pyplot(fig); plt.close(fig)
            else:
                t_rf, t_rms = st.tabs(["Dashboard Random Forest", "Dashboard RMS"])
                with t_rf:
                    fig = buat_dashboard(f, y, sr, d, p, berisik_rf, n_b, n_p, kolom, imp, ref)
                    st.pyplot(fig); plt.close(fig)
                with t_rms:
                    fig = buat_dashboard_rms(y, sr, d, rms_hasil, prm)
                    st.pyplot(fig); plt.close(fig)

            entri = {
                "_audio_id": audio_id, "file": nama_audio,
                "vonis RF": "BERISIK" if berisik_rf else "TIDAK BERISIK",
                "P(RF)": round(p, 3), "pohon_berisik": f"{n_b}/{n_p}",
                "vonis RMS": "BERISIK" if rms_hasil["berisik"] else "TIDAK BERISIK",
                "skor RMS": round(rms_hasil["skor"], 2), "durasi bising %": round(rms_hasil["persen"], 1),
                "sepakat": "ya" if berisik_rf == rms_hasil["berisik"] else "tidak",
            }
            indeks = next((i for i, it in enumerate(st.session_state.riwayat) if it.get("_audio_id") == audio_id), None)
            if indeks is None and audio_baru:
                st.session_state.riwayat.append(entri)
            elif indeks is not None:
                st.session_state.riwayat[indeks] = entri
        elif hasil and hasil["audio_id"] == audio_id:
            st.error(f"Gagal memproses audio: {hasil['error']}")
    else:
        st.session_state.hasil_audio = None
        st.session_state.audio_langsung.reset()
        if sumber_audio == "Unggah berkas":
            st.info("Unggah rekaman (mp3 / wav / m4a / flac / ogg) untuk analisis otomatis.")
        elif sumber_audio == "Rekam langsung":
            st.info("Izinkan akses mikrofon di browser, lalu selesaikan rekaman untuk analisis otomatis.")

with tab_riw:
    if st.session_state.riwayat:
        riwayat = pd.DataFrame(st.session_state.riwayat).drop(columns=["_audio_id"], errors="ignore")
        st.dataframe(riwayat, use_container_width=True)
        if st.button("Hapus riwayat"):
            st.session_state.riwayat = []; st.rerun()
    else:
        st.write("Belum ada pengujian.")
