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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
import secrets
from datetime import datetime
import altair as alt
import streamlit.components.v1 as components
from deteksi_logger import (
    inisialisasi_db, log_deteksi, daftar_sesi, hapus_sesi_mati,
    ambil_riwayat, ambil_status_terkini, ambil_data_grafik, hitung_statistik,
    ambil_riwayat_terfilter, ambil_analisis_mendalam, ambil_daftar_sesi,
    bersihkan_sesi_manual, ambil_status_sistem
)

st.set_page_config(page_title="Deteksi Kebisingan Ruangan", layout="wide")

# Inisialisasi database logging saat aplikasi dimulai
inisialisasi_db()

FRAME_LEN, HOP_LEN, SEED = 2048, 512, 42
FITUR_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fitur_audio_kebisingan.csv")
WARNA = {"berisik": "#ef4444", "tenang": "#10b981", "ragu": "#f59e0b",
         "gelap": "#090909", "panel": "#171717", "teks": "#f5f5f5", "redup": "#a3a3a3"}


def cari_audio_peringatan():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    format_audio = {
        ".mp3": "audio/mpeg",
        ".mpeg": "audio/mpeg",
        ".wav": "audio/wav",
    }
    for ekstensi, format_mime in format_audio.items():
        jalur = os.path.join(base_dir, f"peringatan_jangan_berisik{ekstensi}")
        if os.path.isfile(jalur):
            return jalur, format_mime
    raise FileNotFoundError(
        "File peringatan tidak ditemukan. Sediakan peringatan_jangan_berisik.mp3, "
        "peringatan_jangan_berisik.mpeg, atau peringatan_jangan_berisik.wav."
    )


PERINGATAN_AUDIO, FORMAT_AUDIO_PERINGATAN = cari_audio_peringatan()


# ------------------------------------------------------------------ model
def buat_model(nama_model):
    if nama_model == "KNN":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=13, weights="distance", metric="manhattan")),
        ])
    if nama_model == "SVM":
        return Pipeline([
            ("scaler", RobustScaler()),
            ("classifier", SVC(
                C=2.0, kernel="rbf", probability=True,
                random_state=SEED,
            )),
        ])
    if nama_model == "Random Forest":
        return RandomForestClassifier(
            n_estimators=300, random_state=SEED, n_jobs=-1
        )
    raise ValueError(f"Model klasifikasi tidak dikenal: {nama_model}")


@st.cache_resource(show_spinner="Melatih dan mengevaluasi model dari data fitur…")
def siapkan_model(nama_model):
    df = pd.read_csv(FITUR_CSV)
    kolom = [c for c in df.columns if c not in ("file_name", "label", "grup", "path")]
    X = df[kolom].values
    y = (df["label"] == "Berisik").astype(int).values
    grup = df["grup"].astype(str).values

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    estimator_cv = buat_model(nama_model)
    prediksi_cv = cross_val_predict(estimator_cv, X, y, cv=cv, groups=grup)
    akurasi_cv = accuracy_score(y, prediksi_cv)
    model = buat_model(nama_model).fit(X, y)
    imp = None
    if isinstance(model, RandomForestClassifier):
        imp = pd.Series(model.feature_importances_, index=kolom).sort_values(ascending=False)
    ref = {
        "mu": df[kolom].mean(), "sd": df[kolom].std().replace(0, 1),
        "mu_b": df[df.label == "Berisik"][kolom].mean(),
        "mu_t": df[df.label == "Tidak Berisik"][kolom].mean(),
    }
    return model, kolom, imp, ref, akurasi_cv, len(df)


def prediksi_model(model, x):
    idx_berisik = list(model.classes_).index(1)
    p_berisik = float(model.predict_proba(x)[0][idx_berisik])
    if isinstance(model, RandomForestClassifier):
        suara_pohon = np.array([
            pohon.predict_proba(x)[0][idx_berisik] for pohon in model.estimators_
        ])
        return p_berisik, int((suara_pohon >= .5).sum()), len(suara_pohon)
    return p_berisik, None, None


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


def _hangatkan_numba():
    try:
        dummy_y = np.zeros(FRAME_LEN * 2, dtype=np.float32)
        ekstrak_fitur_sinyal(dummy_y, 48000)
    except Exception:
        pass


_hangatkan_numba()


class PenyanggaAudioLangsung:
    def __init__(self):
        self._lock = threading.Lock()
        self._reset_buffer()
        self.hasil = None
        self.error = None
        self.nomor_hasil = 0
        self.jumlah_frame = 0

    def _reset_buffer(self):
        self._potongan = deque()
        self._jumlah_sampel = 0
        self._sampel_sejak_prediksi = 0
        self._sr = None

    def reset(self):
        with self._lock:
            self._reset_buffer()
            self.hasil = None
            self.error = None
            self.nomor_hasil = 0
            self.jumlah_frame = 0

    def proses_frame(self, frame, model, kolom):
        try:
            data = frame.to_ndarray()
            if np.issubdtype(data.dtype, np.integer):
                batas = max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
                data = data.astype(np.float32) / batas
            else:
                data = data.astype(np.float32, copy=False)
            if data.ndim > 1:
                if data.shape[0] in (1, 2) and data.shape[1] > data.shape[0]:
                    data = data.mean(axis=0)
                else:
                    data = data.mean(axis=-1)
            data = np.nan_to_num(data, copy=False).astype(np.float32).ravel()
            sr = frame.sample_rate
            if not sr or data.size == 0:
                return

            with self._lock:
                if sr != self._sr:
                    self._reset_buffer()
                    self._sr = sr
                self.jumlah_frame += 1
                self._potongan.append(data.copy())
                self._jumlah_sampel += data.size
                self._sampel_sejak_prediksi += data.size

                ukuran_min = int(sr * 1.0)
                ukuran_jendela = int(sr * 2)
                while self._jumlah_sampel > ukuran_jendela:
                    lebih = self._jumlah_sampel - ukuran_jendela
                    awal = self._potongan[0]
                    if lebih >= awal.size:
                        self._potongan.popleft()
                        self._jumlah_sampel -= awal.size
                    else:
                        self._potongan[0] = awal[lebih:]
                        self._jumlah_sampel -= lebih

                if (self._jumlah_sampel < ukuran_min
                        or self._sampel_sejak_prediksi < int(sr * 0.8)):
                    return
                y = np.concatenate(tuple(self._potongan))
                self._sampel_sejak_prediksi = 0

            f, _, _, _ = ekstrak_fitur_sinyal(y, sr)
            hasil = {"fitur": f}
            if model is not None and kolom is not None:
                x = pd.DataFrame([f])[kolom].values.astype(np.float32)
                p, n_berisik, n_pohon = prediksi_model(model, x)
                hasil["p"] = p
                if n_berisik is not None:
                    hasil.update({"n_berisik": n_berisik, "n_pohon": n_pohon})
            with self._lock:
                self.hasil = hasil
                self.error = None
                self.nomor_hasil += 1
        except Exception as e:
            with self._lock:
                self.error = str(e)

    def baca(self):
        with self._lock:
            return self.hasil, self.error, self.nomor_hasil, self.jumlah_frame


def tingkat_keyakinan(p):
    j = abs(p - 0.5)
    return "SANGAT YAKIN" if j >= .35 else "YAKIN" if j >= .20 else "CENDERUNG" if j >= .10 else "RAGU-RAGU"


def tentukan_vonis_model(p):
    return p >= 0.5


def tentukan_vonis_rms(rms_dbfs, ambang_rms_dbfs):
    return rms_dbfs >= ambang_rms_dbfs


def tampilkan_peringatan_sampel(audio_id, metode, berisik):
    if not berisik:
        return
    peringatan_diputar = st.session_state.setdefault("peringatan_rekaman_diputar", set())
    kunci_peringatan = (audio_id, metode)
    autoplay = (
        st.session_state.get("autoplay_peringatan", False)
        and kunci_peringatan not in peringatan_diputar
    )
    st.warning(
        "Peringatan: hasil deteksi BERISIK. "
        "Jika suara tidak terdengar, tekan tombol putar pada pemutar di bawah."
    )
    st.audio(
        PERINGATAN_AUDIO,
        format=FORMAT_AUDIO_PERINGATAN,
        autoplay=autoplay,
    )
    if autoplay:
        peringatan_diputar.add(kunci_peringatan)


def kartu_hasil_model(model_name, nama, p, n_berisik, n_pohon, f, berisik):
    ragu = abs(p - .5) < .10
    warna = WARNA["ragu"] if ragu else WARNA["berisik"] if berisik else WARNA["tenang"]
    teks = "BERISIK" if berisik else "TIDAK BERISIK"
    kotak = lambda k, v, s="": (f"<div style='background:{WARNA['panel']};padding:8px 10px;border-radius:8px'>"
                                 f"<div style='color:{WARNA['redup']};font-size:10px'>{k}</div><b>{v}</b>"
                                 f"<div style='color:{WARNA['redup']};font-size:10px'>{s}</div></div>")
    return f"""
    <div style="background:{WARNA['gelap']};color:{WARNA['teks']};padding:20px 24px;border-radius:14px;
                border-left:8px solid {warna};border:1px solid #334155;border-left:8px solid {warna};font-family:'Segoe UI',Arial,sans-serif">
      <div>
        <div style="font-size:11px;letter-spacing:2px;color:{WARNA['redup']}">{model_name.upper()} · {nama}</div>
        <div style="font-size:32px;font-weight:800;color:{warna};margin-top:4px">{teks}</div>
        <div style="font-size:13px;color:{WARNA['redup']}">Keyakinan model: <b style="color:{WARNA['teks']}">{tingkat_keyakinan(p)}</b></div>
      </div>
      <div style="margin-top:16px">
        <div style="display:flex;justify-content:space-between;font-size:11px;color:{WARNA['redup']}">
          <span>Tenang</span><span>P(berisik) = <b style="color:{WARNA['teks']}">{p:.3f}</b></span><span>Berisik</span></div>
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


def kartu_hasil_rms(nama, f, berisik, ambang_rms_dbfs):
    warna = WARNA["berisik"] if berisik else WARNA["tenang"]
    teks = "BERISIK" if berisik else "TIDAK BERISIK"
    return f"""
    <div style="background:{WARNA['gelap']};color:{WARNA['teks']};padding:20px 24px;border-radius:14px;
                border:1px solid #334155;border-left:8px solid {warna};font-family:'Segoe UI',Arial,sans-serif">
      <div style="font-size:11px;letter-spacing:2px;color:{WARNA['redup']}">METODE RMS · {nama}</div>
      <div style="font-size:32px;font-weight:800;color:{warna};margin-top:4px">{teks}</div>
      <div style="margin-top:10px;font-size:14px">
        RMS: <b>{f['rms_dbfs']:.1f} dBFS</b> · Ambang: <b>{ambang_rms_dbfs:.1f} dBFS</b>
      </div>
      <div style="font-size:12px;color:{WARNA['redup']};margin-top:4px">
        {f['rms_dbfs']:.1f} dBFS {'≥' if berisik else '<'} {ambang_rms_dbfs:.1f} dBFS
      </div>
    </div>"""


def buat_dashboard(f, y, sr, d, p, berisik, n_berisik, n_pohon, kolom, imp, ref, model_name):
    top = list(imp.head(8).index) if imp is not None else list(kolom[:8])
    warna = WARNA["ragu"] if abs(p - .5) < .10 else WARNA["berisik"] if berisik else WARNA["tenang"]
    fig = plt.figure(figsize=(14, 8.2), facecolor=WARNA["gelap"])
    gs = fig.add_gridspec(2, 2, hspace=.42, wspace=.28)

    def gaya(ax, judul):
        ax.set_facecolor(WARNA["panel"]); ax.set_title(judul, color=WARNA["teks"], fontsize=11, loc="left", fontweight="bold")
        ax.tick_params(colors=WARNA["redup"], labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#334155")
        ax.grid(color="#334155", alpha=.5, lw=.6)

    ax1 = fig.add_subplot(gs[0, 0])
    if n_berisik is not None and n_pohon is not None:
        gaya(ax1, f"Voting {n_pohon} pohon")
        jml = [n_pohon - n_berisik, n_berisik]
    else:
        gaya(ax1, f"Probabilitas {model_name}")
        jml = [1 - p, p]
    bars = ax1.bar(["Tidak Berisik", "Berisik"], jml, color=[WARNA["tenang"], WARNA["berisik"]], width=.55)
    for b, j in zip(bars, jml):
        persen = j * 100 if n_pohon is None else j / n_pohon * 100
        ax1.text(b.get_x() + b.get_width() / 2, j + .015, f"{j:.2f} ({persen:.0f}%)" if n_pohon is None else f"{j} ({persen:.0f}%)",
                 ha="center", color=WARNA["teks"], fontsize=11, fontweight="bold")
    if n_pohon is not None:
        ax1.axhline(n_pohon / 2, color="white", ls="--", lw=1, alpha=.7)
        ax1.set_ylim(0, n_pohon * 1.12)
    else:
        ax1.set_ylim(0, 1.12)
    ax1.tick_params(axis="x", labelcolor=WARNA["teks"], labelsize=10)

    ax2 = fig.add_subplot(gs[0, 1])
    judul_profil = "Fitur terpenting RF" if imp is not None else "Profil fitur terpilih"
    gaya(ax2, f"{judul_profil} (z-score vs dataset)")
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
    ax3 = fig.add_subplot(gs[1, 0]); gaya(ax3, "Gelombang & amplop RMS")
    ax3.plot(np.arange(len(y)) / sr, y, lw=.4, color="#64748b")
    ax3.plot(t, d["rms"], color=warna, lw=1.8); ax3.plot(t, -d["rms"], color=warna, lw=1.8)
    ax3.set_xlabel("Waktu (detik)", color=WARNA["redup"], fontsize=9)

    ax4 = fig.add_subplot(gs[1, 1]); gaya(ax4, "Spektrogram & Spectral Centroid")
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=FRAME_LEN, hop_length=HOP_LEN)), ref=np.max)
    librosa.display.specshow(S, sr=sr, hop_length=HOP_LEN, x_axis="time", y_axis="linear", cmap="viridis", ax=ax4)
    ax4.plot(t, d["centroid"], color="white", lw=1.4); ax4.set_ylim(0, min(sr / 2, 12000))
    ax4.set_xlabel("Waktu (detik)", color=WARNA["redup"], fontsize=9); ax4.set_ylabel("Hz", color=WARNA["redup"], fontsize=9)
    return fig


# ------------------------------------------------------------------ UI
st.title("Deteksi Kebisingan Ruangan")
st.write("Pilih metode deteksi, lalu unggah rekaman atau gunakan mikrofon.")
metode_deteksi = st.radio(
    "Metode deteksi",
    ["Model klasifikasi", "RMS"],
    horizontal=True,
    key="metode_deteksi",
)

if metode_deteksi == "Model klasifikasi":
    model_klasifikasi = st.selectbox(
        "Pilih model klasifikasi",
        ["Random Forest", "KNN", "SVM"],
        key="model_klasifikasi",
        help="KNN dan SVM distandardisasi otomatis sebelum prediksi.",
    )
    model, kolom, imp, ref, akurasi_cv, n_data = siapkan_model(model_klasifikasi)
else:
    model_klasifikasi = None
    model, kolom, imp, ref = None, None, None, None

with st.sidebar:
    st.subheader("Suara peringatan")
    st.caption("Aktifkan sekali dan coba suara. Browser mungkin tetap meminta klik putar.")
    if st.button(
        "Aktifkan autoplay",
        disabled=st.session_state.get("autoplay_peringatan", False),
        key="aktifkan_autoplay_peringatan",
    ):
        st.session_state.autoplay_peringatan = True
        st.session_state.uji_autoplay_peringatan = True
    if st.session_state.get("autoplay_peringatan", False):
        st.success("Autoplay peringatan aktif; kebijakan browser tetap dapat membatasinya.")
    if st.session_state.pop("uji_autoplay_peringatan", False):
        st.audio(PERINGATAN_AUDIO, format=FORMAT_AUDIO_PERINGATAN, autoplay=True)

    if metode_deteksi == "Model klasifikasi":
        st.header("Tentang Model")
        st.metric("Akurasi validasi silang", f"{akurasi_cv * 100:.1f}%")
        detail_model = (
            "300 pohon" if model_klasifikasi == "Random Forest"
            else "13 tetangga (Manhattan)" if model_klasifikasi == "KNN"
            else "kernel RBF (RobustScaler)"
        )
        st.caption(f"{model_klasifikasi} · {detail_model} · {len(kolom)} fitur · {n_data} data latih. "
                   "Validasi 5-fold dikelompokkan per mahasiswa.")
        if imp is not None:
            st.markdown("**Fitur terpenting**")
            st.bar_chart(imp.head(8)[::-1])
        elif model_klasifikasi in ("KNN", "SVM"):
            st.caption("Fitur distandardisasi agar skala pengukuran tidak mendominasi model.")
        st.caption("Hasil bersifat indikatif. Rekaman yang sangat berbeda dari data latih bisa salah diklasifikasi.")
    else:
        st.header("Tentang RMS")
        st.write("Metode ini membandingkan level RMS rekaman dengan ambang yang dipilih.")
        st.caption("RMS mengukur level digital audio (dBFS), bukan tingkat tekanan suara akustik (dB SPL).")

if "riwayat" not in st.session_state:
    st.session_state.riwayat = []
if "hasil_audio" not in st.session_state:
    st.session_state.hasil_audio = None
if "audio_langsung" not in st.session_state:
    st.session_state.audio_langsung = PenyanggaAudioLangsung()
if "peringatan_realtime_aktif" not in st.session_state:
    st.session_state.peringatan_realtime_aktif = False

# Daftarkan sesi ini ke DB agar admin bisa pantau jumlah pengguna aktif
_ctx = st.runtime.scriptrunner.get_script_run_ctx()
_sesi_id = _ctx.session_id if _ctx is not None else "sesi_lokal"
daftar_sesi(_sesi_id)
hapus_sesi_mati(batas_detik=90)

konfigurasi_model = (metode_deteksi, model_klasifikasi)
if "konfigurasi_model_aktif" not in st.session_state:
    st.session_state.konfigurasi_model_aktif = konfigurasi_model
elif st.session_state.konfigurasi_model_aktif != konfigurasi_model:
    st.session_state.audio_langsung.reset()
    st.session_state.hasil_audio = None
    st.session_state.peringatan_realtime_aktif = False
    st.session_state.konfigurasi_model_aktif = konfigurasi_model

tab_uji, tab_riw, tab_admin = st.tabs(["Uji Sampel", "Riwayat Sesi", "🔒 Panel Admin"])

with tab_uji:
    if metode_deteksi == "RMS":
        ambang_rms_dbfs = st.slider(
            "Ambang suara keras (RMS, dBFS)",
            min_value=-40.0, max_value=-10.0, value=-30.0, step=0.5,
            help="Rekaman dengan level RMS pada atau di atas ambang diputuskan BERISIK.",
        )
        st.caption("dBFS adalah level digital rekaman, bukan pengukuran dB SPL; hasil dipengaruhi mikrofon, jarak, dan gain.")
        st.info("Mode RMS menilai audio hanya dari ambang RMS; Random Forest tidak digunakan.")
    else:
        st.info(f"Mode {model_klasifikasi} menggunakan model saja; ambang RMS tidak memengaruhi vonis.")
    sumber_audio = st.radio(
        "Sumber audio", ["Unggah berkas", "Rekam langsung", "Deteksi realtime"],
        horizontal=True,
    )
    if sumber_audio == "Unggah berkas":
        audio = st.file_uploader("Pilih berkas audio", type=["mp3", "wav", "m4a", "flac", "ogg"])
        nama_audio = audio.name if audio is not None else ""
    elif sumber_audio == "Rekam langsung":
        audio = st.audio_input("Rekam suara menggunakan mikrofon")
        nama_audio = "Rekaman langsung.wav" if audio is not None else ""

    if sumber_audio == "Deteksi realtime":
        from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

        st.caption("Tekan START dan izinkan mikrofon. Hasil diperbarui setiap detik dari 2 detik suara terakhir.")
        st.session_state.hasil_audio = None
        state_audio = st.session_state.audio_langsung

        def callback_audio(frame):
            if metode_deteksi == "Model klasifikasi":
                state_audio.proses_frame(frame, model, kolom)
            else:
                state_audio.proses_frame(frame, None, None)
            return frame

        # Konfigurasi STUN/TURN agar WebRTC dapat bertukar paket audio di NAT, Docker, & Cloud
        ice_servers = [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun.cloudflare.com:3478"]},
        ]
        try:
            if "turn" in st.secrets:
                ice_servers.append(dict(st.secrets["turn"]))
        except Exception:
            pass

        rtc_config = RTCConfiguration({"iceServers": ice_servers})

        webrtc_ctx = webrtc_streamer(
            key=f"deteksi-kebisingan-realtime-{metode_deteksi}-{model_klasifikasi}",
            mode=WebRtcMode.SENDONLY,
            rtc_configuration=rtc_config,
            audio_frame_callback=callback_audio,
            media_stream_constraints={"audio": True, "video": False},
            sendback_audio=False,
            async_processing=True,
        )

        @st.fragment(run_every="1s")
        def tampilkan_hasil_realtime():
            hasil_live, error_live, nomor_hasil, jumlah_frame = state_audio.baca()
            if not webrtc_ctx.state.playing:
                st.session_state.peringatan_realtime_aktif = False
                st.session_state.peringatan_realtime_autoplay_diputar = False
                st.info("Tekan START untuk mulai mendeteksi melalui mikrofon. Tekan STOP untuk mengakhiri.")
                return
            if error_live:
                st.error(f"Gagal menganalisis audio langsung: {error_live}")
            if hasil_live is None:
                if jumlah_frame:
                    st.info("Audio diterima. Menunggu 1 detik suara untuk prediksi pertama…")
                else:
                    st.warning("Belum menerima audio. Periksa izin mikrofon dan perangkat input di browser.")
                return

            _waktu_sekarang = datetime.now().strftime("%H:%M:%S")
            fitur_live = hasil_live["fitur"]
            if metode_deteksi == "Model klasifikasi":
                p_live = hasil_live["p"]
                berisik_live = tentukan_vonis_model(p_live)
                st.markdown(kartu_hasil_model(
                    model_klasifikasi, "Deteksi realtime", p_live,
                    hasil_live.get("n_berisik"), hasil_live.get("n_pohon"),
                    fitur_live, berisik_live,
                ), unsafe_allow_html=True)
                st.caption(
                    f"🟢 **Live** ({_waktu_sekarang}) · Pembaruan ke-{nomor_hasil} · "
                    f"RMS: {fitur_live['rms_dbfs']:.1f} dBFS · "
                    f"P(berisik): {p_live:.3f} · {model_klasifikasi}"
                )
            else:
                berisik_live = tentukan_vonis_rms(fitur_live["rms_dbfs"], ambang_rms_dbfs)
                st.markdown(kartu_hasil_rms(
                    "Deteksi realtime", fitur_live, berisik_live, ambang_rms_dbfs
                ), unsafe_allow_html=True)
                st.caption(f"🟢 **Live** ({_waktu_sekarang}) · Pembaruan ke-{nomor_hasil} · Mode RMS.")

            # --- Log hasil realtime ke SQLite (hanya saat ada pembaruan hasil baru) ---
            if st.session_state.get("_terakhir_nomor_hasil_dilog") != nomor_hasil:
                st.session_state["_terakhir_nomor_hasil_dilog"] = nomor_hasil
                _met_rt = model_klasifikasi if metode_deteksi == "Model klasifikasi" else "RMS"
                _p_rt = p_live if metode_deteksi == "Model klasifikasi" else None
                _key_rt = tingkat_keyakinan(p_live) if metode_deteksi == "Model klasifikasi" else None
                log_deteksi(
                    sumber="Realtime WebRTC",
                    metode=_met_rt,
                    vonis="BERISIK" if berisik_live else "TIDAK BERISIK",
                    p_berisik=_p_rt,
                    rms_dbfs=fitur_live.get("rms_dbfs"),
                    keyakinan=_key_rt,
                    sesi_id=_sesi_id,
                )

            if berisik_live:
                autoplay = (
                    st.session_state.get("autoplay_peringatan", False)
                    and not st.session_state.get("peringatan_realtime_autoplay_diputar", False)
                )
                st.warning(
                    "Peringatan: hasil deteksi realtime BERISIK. "
                    "Jika suara tidak terdengar, tekan tombol putar pada pemutar di bawah."
                )
                st.audio(
                    PERINGATAN_AUDIO,
                    format=FORMAT_AUDIO_PERINGATAN,
                    autoplay=autoplay,
                )
                if autoplay:
                    st.session_state.peringatan_realtime_autoplay_diputar = True
                st.session_state.peringatan_realtime_aktif = True
            else:
                st.session_state.peringatan_realtime_aktif = False
                st.session_state.peringatan_realtime_autoplay_diputar = False

        tampilkan_hasil_realtime()
        if not webrtc_ctx.state.playing:
            state_audio.reset()
            st.session_state.peringatan_realtime_aktif = False

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
                with st.spinner("Mengekstrak fitur audio…"):
                    f, y, sr, d = ekstrak_fitur(jalur)
                hasil = {
                    "audio_id": audio_id, "nama": nama_audio, "fitur": f, "y": y,
                    "sr": sr, "detail": d,
                }
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
            if metode_deteksi == "Model klasifikasi":
                if "p" not in hasil:
                    x = pd.DataFrame([f])[kolom].values.astype(np.float32)
                    p, n_berisik, n_pohon = prediksi_model(model, x)
                    hasil["p"] = p
                    if n_berisik is not None:
                        hasil.update({"n_berisik": n_berisik, "n_pohon": n_pohon})
                p = hasil["p"]
                n_b, n_p = hasil.get("n_berisik"), hasil.get("n_pohon")
                berisik = tentukan_vonis_model(p)
                st.markdown(kartu_hasil_model(
                    model_klasifikasi, nama_audio, p, n_b, n_p, f, berisik
                ), unsafe_allow_html=True)
                tampilkan_peringatan_sampel(audio_id, model_klasifikasi, berisik)
                st.write("")
                fig = buat_dashboard(
                    f, y, sr, d, p, berisik, n_b, n_p, kolom, imp, ref, model_klasifikasi
                )
                st.pyplot(fig)
                plt.close(fig)
                entri = {
                    "_audio_id": f"{audio_id}:{model_klasifikasi}", "file": nama_audio,
                    "metode": model_klasifikasi,
                    "vonis": "BERISIK" if berisik else "TIDAK BERISIK",
                    "P model (berisik)": round(p, 3),
                    "keyakinan model": tingkat_keyakinan(p),
                }
                if n_b is not None and n_p is not None:
                    entri["pohon_berisik"] = f"{n_b}/{n_p}"
                # --- Log ke SQLite ---
                _sumber_log = "Rekam Langsung" if sumber_audio == "Rekam langsung" else "Upload Berkas"
                log_deteksi(
                    sumber=_sumber_log, metode=model_klasifikasi,
                    vonis="BERISIK" if berisik else "TIDAK BERISIK",
                    p_berisik=p, rms_dbfs=f.get("rms_dbfs"),
                    nama_file=nama_audio, keyakinan=tingkat_keyakinan(p),
                    sesi_id=_sesi_id,
                )
            else:
                berisik = tentukan_vonis_rms(f["rms_dbfs"], ambang_rms_dbfs)
                st.markdown(kartu_hasil_rms(
                    nama_audio, f, berisik, ambang_rms_dbfs
                ), unsafe_allow_html=True)
                tampilkan_peringatan_sampel(audio_id, metode_deteksi, berisik)
                t = librosa.times_like(d["rms"], sr=sr, hop_length=HOP_LEN)
                fig, ax = plt.subplots(figsize=(12, 3), facecolor=WARNA["gelap"])
                ax.set_facecolor(WARNA["panel"])
                ax.plot(np.arange(len(y)) / sr, y, lw=.5, color="#64748b", label="Gelombang audio")
                ax.plot(t, d["rms"], lw=1.8, color=WARNA["berisik"], label="RMS")
                ax.plot(t, -d["rms"], lw=1.8, color=WARNA["berisik"])
                ax.set_title("Gelombang audio dan amplop RMS", color=WARNA["teks"], loc="left")
                ax.set_xlabel("Waktu (detik)", color=WARNA["redup"])
                ax.tick_params(colors=WARNA["redup"])
                ax.legend(facecolor=WARNA["panel"], labelcolor=WARNA["teks"])
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                entri = {
                    "_audio_id": f"{audio_id}:RMS", "file": nama_audio,
                    "metode": "RMS",
                    "vonis": "BERISIK" if berisik else "TIDAK BERISIK",
                    "RMS (dBFS)": round(f["rms_dbfs"], 1),
                    "ambang (dBFS)": ambang_rms_dbfs,
                }
                # --- Log ke SQLite ---
                _sumber_log = "Rekam Langsung" if sumber_audio == "Rekam langsung" else "Upload Berkas"
                log_deteksi(
                    sumber=_sumber_log, metode="RMS",
                    vonis="BERISIK" if berisik else "TIDAK BERISIK",
                    rms_dbfs=f.get("rms_dbfs"), nama_file=nama_audio,
                    sesi_id=_sesi_id,
                )
            indeks = next(
                (i for i, item in enumerate(st.session_state.riwayat)
                 if item.get("_audio_id") == entri["_audio_id"]),
                None,
            )
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


with tab_admin:
    st.subheader("Panel Admin & Monitoring Akustik")
    admin_pass = os.getenv("ADMIN_PASS")
    if not admin_pass:
        try:
            admin_pass = st.secrets.get("ADMIN_PASS", "admin123")
        except Exception:
            admin_pass = "admin123"

    if "admin_terotentikasi" not in st.session_state:
        st.session_state.admin_terotentikasi = False

    if not st.session_state.admin_terotentikasi:
        st.markdown("""
        <div style="background:#131d33;border:1px solid #24324f;border-radius:14px;padding:26px;max-width:480px;margin:30px auto;text-align:center;box-shadow:0 8px 32px rgba(0,0,0,0.4)">
            <div style="font-size:38px;margin-bottom:10px">🔒</div>
            <h3 style="color:#f8fafc;font-size:20px;margin-bottom:6px">Portal Administrator</h3>
            <p style="font-size:12px;color:#94a3b8;margin-bottom:18px">Masukkan kata sandi admin untuk mengakses metrik real-time, audit log, dan analitik mendalam.</p>
        </div>
        """, unsafe_allow_html=True)
        with st.form("form_login_admin"):
            input_pass = st.text_input("Kata Sandi Admin", type="password", placeholder="Masukkan kata sandi admin...")
            tbl_masuk = st.form_submit_button("Masuk ke Panel Admin", use_container_width=True)
            if tbl_masuk:
                if secrets.compare_digest(input_pass.encode("utf-8"), str(admin_pass).encode("utf-8")):
                    st.session_state.admin_terotentikasi = True
                    st.success("Autentikasi berhasil!")
                    st.rerun()
                else:
                    st.error("Kata sandi salah. Akses ditolak.")
    else:
        # Top Action Bar
        kol_mode, kol_refresh, kol_link, kol_keluar = st.columns([3, 1, 1.6, 0.8])
        with kol_mode:
            opsi_tampilan = st.radio(
                "Mode Tampilan Dashboard",
                ["🎨 Tampilan Native Cyber HD", "🌐 Portal Dashboard Standalone (Embedded Live WS)"],
                horizontal=True,
                label_visibility="collapsed"
            )
        with kol_refresh:
            if st.button("🔄 Segarkan", key="btn_refresh_adm", use_container_width=True, help="Muat ulang seluruh metrik & riwayat terbaru"):
                st.rerun()
        with kol_link:
            st.link_button("🚀 Buka Portal (Port 8000)", url="http://localhost:8000", use_container_width=True, help="Buka dashboard standalone di jendela browser terpisah")
        with kol_keluar:
            if st.button("Keluar", key="tbl_keluar_admin", use_container_width=True):
                st.session_state.admin_terotentikasi = False
                st.rerun()

        st.divider()

        # OPSI A: Portal Standalone Embedded
        if opsi_tampilan == "🌐 Portal Dashboard Standalone (Embedded Live WS)":
            embed_url = f"http://localhost:8000/embed?token={admin_pass}"
            components.iframe(embed_url, height=920, scrolling=True)
            st.caption("💡 Tips: Anda juga dapat mengakses portal langsung di tab baru melalui [http://localhost:8000](http://localhost:8000)")

        # OPSI B: Tampilan Native Cyber HD
        else:
            # ── Helpers Desain Cyber ─────────────────────────────────────────
            def _badge_sumber_html(sumber_text):
                s = str(sumber_text or "")
                if "Realtime" in s:
                    return '<span style="background:rgba(6,182,212,0.18);color:#06b6d4;border:1px solid rgba(6,182,212,0.35);padding:3px 9px;border-radius:999px;font-weight:700;font-size:11px;display:inline-block">📡 Realtime</span>'
                elif "Rekam" in s:
                    return '<span style="background:rgba(139,92,246,0.18);color:#8b5cf6;border:1px solid rgba(139,92,246,0.35);padding:3px 9px;border-radius:999px;font-weight:700;font-size:11px;display:inline-block">🎙️ Rekam</span>'
                elif "Upload" in s or "Unggah" in s:
                    return '<span style="background:rgba(59,130,246,0.18);color:#3b82f6;border:1px solid rgba(59,130,246,0.35);padding:3px 9px;border-radius:999px;font-weight:700;font-size:11px;display:inline-block">📁 Upload</span>'
                return f'<span style="background:rgba(148,163,184,0.15);color:#94a3b8;border:1px solid rgba(148,163,184,0.3);padding:3px 9px;border-radius:999px;font-size:11px">{s}</span>'

            def _badge_vonis_html(vonis_text):
                v = str(vonis_text or "")
                if v == "BERISIK":
                    return '<span style="background:rgba(239,68,68,0.18);color:#ef4444;border:1px solid rgba(239,68,68,0.35);padding:3px 9px;border-radius:999px;font-weight:700;font-size:11px;display:inline-block">● BERISIK</span>'
                return '<span style="background:rgba(16,185,129,0.18);color:#10b981;border:1px solid rgba(16,185,129,0.35);padding:3px 9px;border-radius:999px;font-weight:700;font-size:11px;display:inline-block">● TENANG</span>'

            def _kartu_kpi_html(label, val, sub, border_col, val_col):
                return f"""
                <div style="background:#131d33;border:1px solid #24324f;border-left:4px solid {border_col};border-radius:12px;padding:14px 18px;margin-bottom:14px">
                    <div style="font-size:11px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:0.8px">{label}</div>
                    <div style="font-size:30px;font-weight:900;color:{val_col};margin:4px 0;line-height:1.2">{val}</div>
                    <div style="font-size:11px;color:#64748b">{sub}</div>
                </div>
                """

            def _hero_status_html(terkini_data):
                if not terkini_data:
                    return '<div style="background:#131d33;border:1px solid #24324f;border-radius:14px;padding:24px;text-align:center;color:#94a3b8">Belum ada data deteksi di basis data.</div>'
                is_berisik = terkini_data.get("vonis") == "BERISIK"
                w_vonis = "#ef4444" if is_berisik else "#10b981"
                bg_grad = "rgba(239,68,68,0.12)" if is_berisik else "rgba(16,185,129,0.12)"
                shadow = "rgba(239,68,68,0.6)" if is_berisik else "rgba(16,185,129,0.4)"

                raw_db = terkini_data.get("rms_dbfs")
                db_val = float(raw_db) if raw_db is not None else -60.0
                pct = max(0.0, min(100.0, ((db_val + 60.0) / 60.0) * 100.0))
                vu_col = "#ef4444" if db_val >= -20.0 else "#f59e0b" if db_val >= -30.0 else "#10b981"

                badge_sumber = _badge_sumber_html(terkini_data.get("sumber"))
                keyakinan = terkini_data.get("keyakinan") or "STANDAR"
                p_str = f" · P={terkini_data['p_berisik']:.3f}" if terkini_data.get("p_berisik") is not None else ""
                file_str = f" · File: {terkini_data.get('nama_file')}" if terkini_data.get("nama_file") else ""

                return f"""
                <div style="background:linear-gradient(90deg, {bg_grad} 0%, #131d33 70%);border:1px solid #24324f;border-left:8px solid {w_vonis};border-radius:14px;padding:22px 26px;margin-bottom:20px">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start">
                        <div>
                            <div style="font-size:11px;letter-spacing:1.5px;color:#94a3b8;font-weight:800;text-transform:uppercase">STATUS AKUSTIK TERKINI</div>
                            <div style="font-size:38px;font-weight:900;color:{w_vonis};text-shadow:0 0 20px {shadow};line-height:1.2;margin:4px 0">{terkini_data.get('vonis')}</div>
                            <div style="font-size:12px;color:#cbd5e1;display:flex;align-items:center;gap:8px;flex-wrap:wrap">
                                <span>{terkini_data.get('timestamp')}</span> · 
                                {badge_sumber} · 
                                <span><b>{terkini_data.get('metode')}</b>{p_str}{file_str}</span>
                            </div>
                        </div>
                        <div style="text-align:right;display:flex;flex-direction:column;align-items:flex-end;gap:8px">
                            <div style="width:18px;height:18px;border-radius:50%;background:{w_vonis};box-shadow:0 0 16px {shadow}"></div>
                            <div style="background:#070d1a;padding:6px 12px;border-radius:8px;border:1px solid #24324f;font-size:11px;color:#94a3b8">
                                KEYAKINAN: <b style="color:#f8fafc">{keyakinan}</b>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Digital VU Meter -->
                    <div style="margin-top:16px;max-width:560px">
                        <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:4px">
                            <span style="color:#94a3b8;font-weight:700">Digital VU Meter (Level Suara)</span>
                            <b style="color:#f8fafc">{db_val:.1f} dBFS</b>
                        </div>
                        <div style="height:14px;background:#070d1a;border-radius:7px;border:1px solid #24324f;overflow:hidden">
                            <div style="width:{pct:.1f}%;height:100%;background:{vu_col};border-radius:7px;transition:width 0.3s"></div>
                        </div>
                        <div style="display:flex;justify-content:space-between;font-size:10px;color:#64748b;margin-top:4px">
                            <span>-60 dB (Sunyi)</span>
                            <span>-30 dB (Wajar)</span>
                            <span>-20 dB (Batas Bising)</span>
                            <span>0 dB (Keras)</span>
                        </div>
                    </div>
                </div>
                """

            sub_live, sub_analisis, sub_riwayat, sub_sesi, sub_sistem = st.tabs([
                "📡 Live Monitoring",
                "📊 Analisis & Statistik",
                "📋 Audit Log & Ekspor",
                "👥 Sesi Pengguna",
                "⚙️ Status Sistem & Model"
            ])

            # ── 1. SUB-TAB: LIVE MONITORING ──────────────────────────────────
            with sub_live:
                stat = hitung_statistik()
                terkini = ambil_status_terkini()

                # Status Hero
                st.markdown(_hero_status_html(terkini), unsafe_allow_html=True)

                # 5 Metrik Ringkasan Utama
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1:
                    st.markdown(_kartu_kpi_html("Total Deteksi", stat.get("total", 0), "Keseluruhan sesi", "#3b82f6", "#3b82f6"), unsafe_allow_html=True)
                with c2:
                    st.markdown(_kartu_kpi_html("Vonis Berisik", stat.get("berisik", 0), f"{stat.get('persen_berisik', 0)}% rasio", "#ef4444", "#ef4444"), unsafe_allow_html=True)
                with c3:
                    st.markdown(_kartu_kpi_html("Vonis Tenang", stat.get("tenang", 0), "Kondisi kondusif", "#10b981", "#10b981"), unsafe_allow_html=True)
                with c4:
                    st.markdown(_kartu_kpi_html("Hari Ini", stat.get("hari_ini", 0), f"{stat.get('berisik_hari_ini', 0)} bising", "#f59e0b", "#f59e0b"), unsafe_allow_html=True)
                with c5:
                    st.markdown(_kartu_kpi_html("Klien Aktif", stat.get("sesi_aktif", 0), "Sesi online aktif", "#06b6d4", "#06b6d4"), unsafe_allow_html=True)

                # 3 Metrik Sumber Audio
                st.markdown('<div style="font-size:11px;font-weight:800;color:#94a3b8;text-transform:uppercase;letter-spacing:0.8px;margin:8px 0 10px 0">DISTRIBUSI DATA SUMBER AUDIO</div>', unsafe_allow_html=True)
                s1, s2, s3 = st.columns(3)
                with s1:
                    st.markdown(_kartu_kpi_html("📡 Realtime WebRTC", stat.get("total_realtime", 0), "Streaming mikrofon langsung", "#06b6d4", "#06b6d4"), unsafe_allow_html=True)
                with s2:
                    st.markdown(_kartu_kpi_html("🎙️ Rekam Langsung", stat.get("total_rekam", 0), "Audio input rekaman suara", "#8b5cf6", "#8b5cf6"), unsafe_allow_html=True)
                with s3:
                    st.markdown(_kartu_kpi_html("📁 Upload Berkas", stat.get("total_upload", 0), "Berkas rekaman diunggah", "#3b82f6", "#3b82f6"), unsafe_allow_html=True)

                # Grafik Live RMS Altair
                st.markdown("#### 📈 Aliran Level Suara (dBFS) — 15 Menit Terakhir")
                data_grafik = ambil_data_grafik(menit=15)
                if data_grafik:
                    df_g = pd.DataFrame(data_grafik)
                    df_g["waktu"] = df_g["timestamp"].apply(lambda x: str(x).split(" ")[-1] if " " in str(x) else str(x))
                    df_g["rms"] = df_g["rms_dbfs"].fillna(-60.0).round(1)

                    base_ch = alt.Chart(df_g).encode(
                        x=alt.X("waktu:N", title="Waktu", axis=alt.Axis(labelColor="#94a3b8", titleColor="#94a3b8", labelAngle=-45)),
                        y=alt.Y("rms:Q", title="Level RMS (dBFS)", scale=alt.Scale(domain=[-60, 0]), axis=alt.Axis(labelColor="#94a3b8", titleColor="#94a3b8", gridColor="#1e293b"))
                    )
                    line_ch = base_ch.mark_line(color="#3b82f6", strokeWidth=2, interpolate="monotone")
                    area_ch = base_ch.mark_area(color="#3b82f6", opacity=0.15, interpolate="monotone")
                    dots_ch = base_ch.mark_circle(size=60).encode(
                        color=alt.Color("vonis:N", scale=alt.Scale(domain=["BERISIK", "TIDAK BERISIK"], range=["#ef4444", "#10b981"]), legend=alt.Legend(title="Vonis", labelColor="#f8fafc", titleColor="#94a3b8")),
                        tooltip=[alt.Tooltip("timestamp:N", title="Waktu"), alt.Tooltip("rms:Q", title="RMS (dBFS)"), alt.Tooltip("vonis:N", title="Vonis")]
                    )
                    rule_ch = alt.Chart(pd.DataFrame({"y": [-20]})).mark_rule(color="#ef4444", strokeDash=[6, 4], size=1.5).encode(y="y:Q")
                    chart_full = (area_ch + line_ch + dots_ch + rule_ch).properties(height=260).configure_view(strokeWidth=0).configure(background="#070d1a")
                    st.altair_chart(chart_full, use_container_width=True)
                else:
                    st.info("Belum ada aliran data suara dalam 15 menit terakhir.")

            # ── 2. SUB-TAB: ANALISIS & STATISTIK ─────────────────────────────
            with sub_analisis:
                analisis = ambil_analisis_mendalam()
                eks = analisis.get("ekstrem", {})

                # 3 Kartu Ekstrem
                e1, e2, e3 = st.columns(3)
                with e1:
                    st.markdown(_kartu_kpi_html("RMS Terkeras (Peak)", f"{eks.get('max_rms', '-')} dBFS", "Puncak energi sinyal", "#ef4444", "#ef4444"), unsafe_allow_html=True)
                with e2:
                    st.markdown(_kartu_kpi_html("RMS Rata-Rata", f"{eks.get('avg_rms', '-')} dBFS", "Tingkat kebisingan umum", "#3b82f6", "#3b82f6"), unsafe_allow_html=True)
                with e3:
                    st.markdown(_kartu_kpi_html("RMS Paling Hening", f"{eks.get('min_rms', '-')} dBFS", "Noise floor dasar", "#10b981", "#10b981"), unsafe_allow_html=True)

                st.divider()

                # Dua Grafik Analisis Altair
                ca1, ca2 = st.columns(2)
                with ca1:
                    st.markdown("##### ⏰ Pola Kebisingan Berdasarkan Jam")
                    jam_data = analisis.get("per_jam", [])
                    if jam_data:
                        df_jam = pd.DataFrame(jam_data)
                        df_jam["jam_label"] = df_jam["jam"].apply(lambda j: f"{j}:00")
                        ch_jam = alt.Chart(df_jam).transform_fold(
                            ["berisik", "total"], as_=["Kategori", "Jumlah"]
                        ).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                            x=alt.X("jam_label:N", title="Jam"),
                            y=alt.Y("Jumlah:Q", title="Frekuensi"),
                            color=alt.Color("Kategori:N", scale=alt.Scale(domain=["berisik", "total"], range=["#ef4444", "#3b82f6"])),
                            xOffset="Kategori:N"
                        ).properties(height=260).configure_view(strokeWidth=0).configure(background="#070d1a")
                        st.altair_chart(ch_jam, use_container_width=True)
                    else:
                        st.caption("Belum ada data rekaman per jam.")

                with ca2:
                    st.markdown("##### 🤖 Komparasi Metode Klasifikasi")
                    met_data = analisis.get("per_metode", [])
                    if met_data:
                        df_met = pd.DataFrame(met_data)
                        ch_met = alt.Chart(df_met).transform_fold(
                            ["berisik", "tenang"], as_=["Status", "Jumlah"]
                        ).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                            x=alt.X("metode:N", title="Metode"),
                            y=alt.Y("Jumlah:Q", title="Jumlah Deteksi"),
                            color=alt.Color("Status:N", scale=alt.Scale(domain=["berisik", "tenang"], range=["#ef4444", "#10b981"])),
                            xOffset="Status:N"
                        ).properties(height=260).configure_view(strokeWidth=0).configure(background="#070d1a")
                        st.altair_chart(ch_met, use_container_width=True)
                    else:
                        st.caption("Belum ada data metode.")

                st.divider()

                # Komparasi Sumber Audio
                st.markdown("##### 🎙️ Komparasi Karakteristik Sumber Audio")
                sum_data = analisis.get("per_sumber", [])
                if sum_data:
                    df_sum = pd.DataFrame(sum_data)
                    ch_sum = alt.Chart(df_sum).transform_fold(
                        ["berisik", "tenang"], as_=["Status", "Jumlah"]
                    ).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                        x=alt.X("sumber:N", title="Sumber Audio"),
                        y=alt.Y("Jumlah:Q", title="Jumlah Deteksi"),
                        color=alt.Color("Status:N", scale=alt.Scale(domain=["berisik", "tenang"], range=["#ef4444", "#10b981"])),
                        xOffset="Status:N"
                    ).properties(height=240).configure_view(strokeWidth=0).configure(background="#070d1a")
                    st.altair_chart(ch_sum, use_container_width=True)

                    # Tabel Karakteristik Sumber
                    rows_html = ""
                    for r in sum_data:
                        total_cnt = r.get("total", 0)
                        berisik_cnt = r.get("berisik", 0)
                        tenang_cnt = r.get("tenang", total_cnt - berisik_cnt)
                        rasio = f"{berisik_cnt / total_cnt * 100:.1f}%" if total_cnt else "0%"
                        avg_r = f"{r['avg_rms']:.1f} dBFS" if r.get("avg_rms") is not None else "—"
                        rows_html += f"""
                        <tr>
                            <td style="padding:10px 14px;border-bottom:1px solid #24324f">{_badge_sumber_html(r.get('sumber'))}</td>
                            <td style="padding:10px 14px;border-bottom:1px solid #24324f"><b>{total_cnt}</b></td>
                            <td style="padding:10px 14px;border-bottom:1px solid #24324f;color:#ef4444;font-weight:700">{berisik_cnt}</td>
                            <td style="padding:10px 14px;border-bottom:1px solid #24324f;color:#10b981;font-weight:700">{tenang_cnt}</td>
                            <td style="padding:10px 14px;border-bottom:1px solid #24324f"><b>{rasio}</b></td>
                            <td style="padding:10px 14px;border-bottom:1px solid #24324f">{avg_r}</td>
                        </tr>
                        """
                    table_html = f"""
                    <div style="border:1px solid #24324f;border-radius:12px;overflow:hidden;margin-top:12px">
                        <table style="width:100%;border-collapse:collapse;font-size:12px;text-align:left;background:#131d33">
                            <thead style="background:#0e1628;color:#94a3b8;font-size:11px;text-transform:uppercase">
                                <tr>
                                    <th style="padding:12px 14px">Sumber Audio</th>
                                    <th style="padding:12px 14px">Total Pengujian</th>
                                    <th style="padding:12px 14px">Vonis Berisik</th>
                                    <th style="padding:12px 14px">Vonis Tenang</th>
                                    <th style="padding:12px 14px">Rasio Berisik</th>
                                    <th style="padding:12px 14px">Rata-rata RMS</th>
                                </tr>
                            </thead>
                            <tbody>{rows_html}</tbody>
                        </table>
                    </div>
                    """
                    st.markdown(table_html, unsafe_allow_html=True)
                else:
                    st.caption("Belum ada data sumber audio.")

            # ── 3. SUB-TAB: AUDIT LOG & EKSPOR ───────────────────────────────
            with sub_riwayat:
                f1, f2, f3, f4, f5 = st.columns([1.2, 1, 1, 1.5, 0.8])
                with f1:
                    f_sumber = st.selectbox("Sumber", ["Semua", "Realtime WebRTC", "Rekam Langsung", "Upload Berkas"], key="adm_f_sumber_hd")
                with f2:
                    f_vonis = st.selectbox("Vonis", ["Semua", "BERISIK", "TIDAK BERISIK"], key="adm_f_vonis_hd")
                with f3:
                    f_metode = st.selectbox("Metode", ["Semua", "Random Forest", "KNN", "SVM", "RMS"], key="adm_f_metode_hd")
                with f4:
                    f_q = st.text_input("Cari berkas / sesi", placeholder="Kata kunci...", key="adm_f_q_hd")
                with f5:
                    f_lim = st.selectbox("Baris", [25, 50, 100], index=1, key="adm_f_lim_hd")

                data_riwayat = ambil_riwayat_terfilter(limit=f_lim, vonis=f_vonis, metode=f_metode, query=f_q, sumber=f_sumber)

                if data_riwayat:
                    rows_riw_html = ""
                    for r in data_riwayat:
                        p_str = f"{r['p_berisik']:.3f}" if r.get("p_berisik") is not None else "—"
                        rms_str = f"<b>{r['rms_dbfs']:.1f} dBFS</b>" if r.get("rms_dbfs") is not None else "—"
                        sesi_str = f"<code>{r['sesi_id'][:8]}…</code>" if r.get("sesi_id") else "—"
                        file_str = r.get("nama_file") or "—"
                        rows_riw_html += f"""
                        <tr>
                            <td style="padding:10px 12px;border-bottom:1px solid #24324f;color:#64748b;font-weight:700">{r.get('id')}</td>
                            <td style="padding:10px 12px;border-bottom:1px solid #24324f;color:#cbd5e1;font-size:11px">{r.get('timestamp')}</td>
                            <td style="padding:10px 12px;border-bottom:1px solid #24324f">{_badge_sumber_html(r.get('sumber'))}</td>
                            <td style="padding:10px 12px;border-bottom:1px solid #24324f"><b>{r.get('metode')}</b></td>
                            <td style="padding:10px 12px;border-bottom:1px solid #24324f">{_badge_vonis_html(r.get('vonis'))}</td>
                            <td style="padding:10px 12px;border-bottom:1px solid #24324f">{p_str}</td>
                            <td style="padding:10px 12px;border-bottom:1px solid #24324f">{rms_str}</td>
                            <td style="padding:10px 12px;border-bottom:1px solid #24324f;color:#94a3b8;font-size:11px">{r.get('keyakinan') or '—'}</td>
                            <td style="padding:10px 12px;border-bottom:1px solid #24324f;color:#94a3b8;font-size:11px">{file_str}</td>
                            <td style="padding:10px 12px;border-bottom:1px solid #24324f">{sesi_str}</td>
                        </tr>
                        """
                    table_riw = f"""
                    <div style="border:1px solid #24324f;border-radius:12px;overflow-x:auto;margin-top:12px">
                        <table style="width:100%;border-collapse:collapse;font-size:12px;text-align:left;background:#131d33">
                            <thead style="background:#0e1628;color:#94a3b8;font-size:11px;text-transform:uppercase">
                                <tr>
                                    <th style="padding:12px 14px">ID</th>
                                    <th style="padding:12px 14px">Waktu</th>
                                    <th style="padding:12px 14px">Sumber</th>
                                    <th style="padding:12px 14px">Metode</th>
                                    <th style="padding:12px 14px">Vonis</th>
                                    <th style="padding:12px 14px">P(Bising)</th>
                                    <th style="padding:12px 14px">RMS dBFS</th>
                                    <th style="padding:12px 14px">Keyakinan</th>
                                    <th style="padding:12px 14px">Berkas</th>
                                    <th style="padding:12px 14px">Sesi ID</th>
                                </tr>
                            </thead>
                            <tbody>{rows_riw_html}</tbody>
                        </table>
                    </div>
                    """
                    st.markdown(table_riw, unsafe_allow_html=True)

                    # Tombol unduh CSV
                    df_adm_csv = pd.DataFrame(data_riwayat)
                    csv_bytes = df_adm_csv.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Unduh Audit Log Terfilter (CSV)",
                        data=csv_bytes,
                        file_name=f"audit_kebisingan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("Tidak ada data riwayat yang cocok dengan filter yang dipilih.")

            # ── 4. SUB-TAB: SESI PENGGUNA ─────────────────────────────────────
            with sub_sesi:
                kol_s1, kol_s2 = st.columns([4, 1.2])
                with kol_s1:
                    st.markdown("#### 👥 Sesi Klien Streamlit Aktif")
                    st.caption("Klien yang aktif mengirim sinyal dalam 60 detik terakhir ditandai Online.")
                with kol_s2:
                    if st.button("🧹 Bersihkan Sesi Mati", key="btn_bersihkan_sesi_hd", use_container_width=True):
                        bersihkan_sesi_manual()
                        st.success("Sesi tidak aktif berhasil dibersihkan.")
                        st.rerun()

                sesi_list = ambil_daftar_sesi()
                if sesi_list:
                    rows_sesi_html = ""
                    for s in sesi_list:
                        status_s = s.get("status", "Offline")
                        if status_s == "Online":
                            badge_s = '<span style="background:rgba(16,185,129,0.18);color:#10b981;border:1px solid rgba(16,185,129,0.35);padding:2px 8px;border-radius:999px;font-weight:700;font-size:11px">● Online</span>'
                        elif status_s == "Idle":
                            badge_s = '<span style="background:rgba(245,158,11,0.18);color:#f59e0b;border:1px solid rgba(245,158,11,0.35);padding:2px 8px;border-radius:999px;font-weight:700;font-size:11px">● Idle</span>'
                        else:
                            badge_s = '<span style="background:rgba(148,163,184,0.15);color:#94a3b8;border:1px solid rgba(148,163,184,0.3);padding:2px 8px;border-radius:999px;font-size:11px">● Offline</span>'

                        rows_sesi_html += f"""
                        <tr>
                            <td style="padding:10px 14px;border-bottom:1px solid #24324f">{badge_s}</td>
                            <td style="padding:10px 14px;border-bottom:1px solid #24324f;font-family:monospace;font-weight:700">{s.get('sesi_id')}</td>
                            <td style="padding:10px 14px;border-bottom:1px solid #24324f;font-size:11px;color:#cbd5e1">{s.get('mulai')}</td>
                            <td style="padding:10px 14px;border-bottom:1px solid #24324f;font-size:11px;color:#cbd5e1">{s.get('terakhir')}</td>
                            <td style="padding:10px 14px;border-bottom:1px solid #24324f;font-size:11px;color:#94a3b8">{s.get('detik_lalu', 0)} detik lalu</td>
                        </tr>
                        """
                    table_sesi = f"""
                    <div style="border:1px solid #24324f;border-radius:12px;overflow:hidden;margin-top:12px">
                        <table style="width:100%;border-collapse:collapse;font-size:12px;text-align:left;background:#131d33">
                            <thead style="background:#0e1628;color:#94a3b8;font-size:11px;text-transform:uppercase">
                                <tr>
                                    <th style="padding:12px 14px">Status</th>
                                    <th style="padding:12px 14px">Session ID</th>
                                    <th style="padding:12px 14px">Waktu Mulai</th>
                                    <th style="padding:12px 14px">Aktivitas Terakhir</th>
                                    <th style="padding:12px 14px">Terakhir Dilihat</th>
                                </tr>
                            </thead>
                            <tbody>{rows_sesi_html}</tbody>
                        </table>
                    </div>
                    """
                    st.markdown(table_sesi, unsafe_allow_html=True)
                else:
                    st.caption("Belum ada sesi tercatat di database.")

            # ── 5. SUB-TAB: STATUS SISTEM & MODEL ────────────────────────────
            with sub_sistem:
                sis = ambil_status_sistem()
                col_m, col_db = st.columns(2)

                with col_m:
                    st.markdown("##### 🤖 Model Machine Learning Aktif")
                    st.markdown("""
                    <div style="display:flex;flex-direction:column;gap:12px">
                        <div style="background:#131d33;border:1px solid #24324f;border-left:4px solid #3b82f6;border-radius:10px;padding:14px">
                            <div style="font-weight:800;color:#3b82f6;font-size:14px">🌲 Random Forest Classifier</div>
                            <div style="color:#94a3b8;font-size:11px;margin-top:3px">Akurasi Validasi Silang: <b>84.34%</b> · 300 Pohon · 25 Fitur Akustik</div>
                        </div>
                        <div style="background:#131d33;border:1px solid #24324f;border-left:4px solid #10b981;border-radius:10px;padding:14px">
                            <div style="font-weight:800;color:#10b981;font-size:14px">📍 K-Nearest Neighbors (KNN)</div>
                            <div style="color:#94a3b8;font-size:11px;margin-top:3px">Akurasi Validasi Silang: <b>80.81%</b> · k=13 · Jarak Manhattan (L1) · Distance Weighted</div>
                        </div>
                        <div style="background:#131d33;border:1px solid #24324f;border-left:4px solid #8b5cf6;border-radius:10px;padding:14px">
                            <div style="font-weight:800;color:#8b5cf6;font-size:14px">🎯 Support Vector Machine (SVM)</div>
                            <div style="color:#94a3b8;font-size:11px;margin-top:3px">Akurasi Validasi Silang: <b>78.79%</b> · C=2.0 · Kernel RBF · RobustScaler (IQR)</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_db:
                    st.markdown("##### 🗄️ Kesehatan Server & Database")
                    db_path_val = sis.get("db_path", "—")
                    db_size_val = f"{sis.get('db_size_kb', 0)} KB"
                    db_rows_val = f"{sis.get('total_deteksi', 0)} baris"
                    st.markdown(f"""
                    <div style="background:#131d33;border:1px solid #24324f;border-radius:12px;padding:16px;display:flex;flex-direction:column;gap:10px;font-size:13px">
                        <div style="display:flex;justify-content:space-between;border-bottom:1px solid #24324f;padding-bottom:8px">
                            <span style="color:#94a3b8">Status Server:</span>
                            <span style="color:#10b981;font-weight:700">🟢 Running (Healthy)</span>
                        </div>
                        <div style="display:flex;justify-content:space-between;border-bottom:1px solid #24324f;padding-bottom:8px">
                            <span style="color:#94a3b8">Database SQLite:</span>
                            <span style="font-family:monospace;font-size:11px;color:#f8fafc">{db_path_val}</span>
                        </div>
                        <div style="display:flex;justify-content:space-between;border-bottom:1px solid #24324f;padding-bottom:8px">
                            <span style="color:#94a3b8">Ukuran Database:</span>
                            <b style="color:#f8fafc">{db_size_val}</b>
                        </div>
                        <div style="display:flex;justify-content:space-between;border-bottom:1px solid #24324f;padding-bottom:8px">
                            <span style="color:#94a3b8">Total Rekaman Log:</span>
                            <b style="color:#f8fafc">{db_rows_val}</b>
                        </div>
                        <div style="display:flex;justify-content:space-between">
                            <span style="color:#94a3b8">Protokol Realtime:</span>
                            <span style="color:#06b6d4;font-weight:700">WebSocket + HTTP Fallback</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.divider()

                # Regulasi Standar Kebisingan
                st.markdown("""
                <div style="background:#131d33;border:1px solid #24324f;border-radius:12px;padding:18px">
                    <h5 style="color:#f8fafc;margin-bottom:8px">📜 Regulasi Ambang Batas Kebisingan Ruangan</h5>
                    <div style="font-size:12px;line-height:1.7;color:#94a3b8">
                        Berdasarkan <b>Permenaker No. 5 Tahun 2018</b> dan <b>Kepmen Lingkungan Hidup No. 48 Tahun 1996</b>:
                        <ul style="margin-left:18px;margin-top:6px">
                            <li><b style="color:#10b981">Zona Sunyi / Kondusif (< -30 dBFS / < 55 dBA)</b>: Rekomendasi ideal untuk ruang perpustakaan, kantor tenang, ruang belajar, dan lab komputer.</li>
                            <li><b style="color:#f59e0b">Zona Waspada (-30 s.d. -20 dBFS / 55 - 70 dBA)</b>: Batas percakapan aktif & diskusi kerja tim. Mulai berpotensi memecah fokus konsentrasi tinggi.</li>
                            <li><b style="color:#ef4444">Zona Berisik (> -20 dBFS / > 85 dBA)</b>: Ambang kebisingan kritis yang memicu alarm suara dan peringatan visual mitigasi kebisingan.</li>
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)
