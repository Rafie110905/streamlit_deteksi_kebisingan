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
                _sumber_log = "Rekam Langsung" if sumber_audio == "Rekam langsung" else "Upload"
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
                _sumber_log = "Rekam Langsung" if sumber_audio == "Rekam langsung" else "Upload"
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
    st.subheader("Panel Admin & Monitoring")
    admin_pass = os.getenv("ADMIN_PASS")
    if not admin_pass:
        try:
            admin_pass = st.secrets.get("ADMIN_PASS", "admin123")
        except Exception:
            admin_pass = "admin123"

    if "admin_terotentikasi" not in st.session_state:
        st.session_state.admin_terotentikasi = False

    if not st.session_state.admin_terotentikasi:
        st.caption("Akses dibatasi khusus administrator.")
        with st.form("form_login_admin"):
            input_pass = st.text_input("Kata Sandi Admin", type="password", placeholder="Masukkan kata sandi admin...")
            tbl_masuk = st.form_submit_button("Masuk ke Panel Admin")
            if tbl_masuk:
                if secrets.compare_digest(input_pass.encode("utf-8"), str(admin_pass).encode("utf-8")):
                    st.session_state.admin_terotentikasi = True
                    st.success("Autentikasi berhasil!")
                    st.rerun()
                else:
                    st.error("Kata sandi salah. Akses ditolak.")
    else:
        kol_judul, kol_link, kol_keluar = st.columns([4, 2, 1])
        with kol_judul:
            st.success("Terkoneksi sebagai Administrator.")
        with kol_link:
            st.link_button("🚀 Buka Dashboard Web (Port 8000)", url="http://localhost:8000", help="Buka dashboard standalone realtime di tab browser baru")
        with kol_keluar:
            if st.button("Keluar", key="tbl_keluar_admin"):
                st.session_state.admin_terotentikasi = False
                st.rerun()

        st.divider()

        sub_live, sub_analisis, sub_riwayat, sub_sesi, sub_sistem = st.tabs([
            "📡 Live Monitoring",
            "📊 Analisis & Statistik",
            "📋 Audit Log & Ekspor",
            "👥 Sesi Pengguna",
            "⚙️ Status Sistem & Model"
        ])

        with sub_live:
            stat = hitung_statistik()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Pengguna Aktif (Sesi)", stat.get("sesi_aktif", 0))
            m2.metric("Total Deteksi", stat.get("total", 0))
            m3.metric("Terdeteksi Berisik", f"{stat.get('berisik', 0)} ({stat.get('persen_berisik', 0)}%)")
            m4.metric("Deteksi Hari Ini", f"{stat.get('hari_ini', 0)} ({stat.get('berisik_hari_ini', 0)} bising)")

            st.markdown("#### Status Deteksi Terkini")
            terkini = ambil_status_terkini()
            if terkini:
                warna_vonis = WARNA["berisik"] if terkini["vonis"] == "BERISIK" else WARNA["tenang"]
                st.markdown(
                    f"""
                    <div style="background:{WARNA['panel']};border-left:6px solid {warna_vonis};padding:14px 18px;border-radius:8px;margin-bottom:15px">
                        <div style="font-size:12px;color:{WARNA['redup']}">WAKTU: {terkini['timestamp']} · SUMBER: {terkini['sumber']} · METODE: {terkini['metode']}</div>
                        <div style="font-size:24px;font-weight:700;color:{warna_vonis};margin:4px 0">{terkini['vonis']}</div>
                        <div style="font-size:13px;color:{WARNA['teks']}">
                            Level RMS: <b>{f"{terkini['rms_dbfs']:.1f} dBFS" if terkini['rms_dbfs'] is not None else "-"}</b> |
                            Probabilitas: <b>{f"{terkini['p_berisik']:.3f}" if terkini['p_berisik'] is not None else "-"}</b> |
                            Berkas: <i>{terkini['nama_file'] or 'Mikrofon / Realtime'}</i>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.info("Belum ada data deteksi yang tercatat di database.")

            st.markdown("#### Grafik Level Suara (RMS dBFS) — 15 Menit Terakhir")
            data_grafik = ambil_data_grafik(menit=15)
            if data_grafik:
                df_grafik = pd.DataFrame(data_grafik)
                if "timestamp" in df_grafik.columns and "rms_dbfs" in df_grafik.columns:
                    df_grafik["waktu"] = df_grafik["timestamp"].apply(lambda x: x.split(" ")[-1] if " " in str(x) else str(x))
                    st.line_chart(df_grafik.set_index("waktu")["rms_dbfs"], width="stretch")
            else:
                st.caption("Belum ada data level suara dalam 15 menit terakhir.")

        with sub_analisis:
            analisis = ambil_analisis_mendalam()
            eks = analisis.get("ekstrem", {})
            e1, e2, e3 = st.columns(3)
            e1.metric("Peak RMS Terkeras", f"{eks.get('max_rms', '-')} dBFS")
            e2.metric("Rata-rata RMS", f"{eks.get('avg_rms', '-')} dBFS")
            e3.metric("Noise Floor Terendah", f"{eks.get('min_rms', '-')} dBFS")

            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### ⏰ Pola Kebisingan per Jam (24 Jam)")
                jam_data = analisis.get("per_jam", [])
                if jam_data:
                    df_jam = pd.DataFrame(jam_data).set_index("jam")
                    st.bar_chart(df_jam[["berisik", "total"]], width="stretch")
                else:
                    st.caption("Belum ada data per jam.")
            with c2:
                st.markdown("##### 🤖 Komparasi Metode Klasifikasi")
                met_data = analisis.get("per_metode", [])
                if met_data:
                    df_met = pd.DataFrame(met_data).set_index("metode")
                    st.bar_chart(df_met[["berisik", "tenang"]], width="stretch")
                else:
                    st.caption("Belum ada data metode.")

        with sub_riwayat:
            f1, f2, f3 = st.columns([1, 1, 2])
            with f1:
                f_vonis = st.selectbox("Filter Vonis", ["Semua", "BERISIK", "TIDAK BERISIK"], key="adm_f_vonis")
            with f2:
                f_metode = st.selectbox("Filter Metode", ["Semua", "Random Forest", "KNN", "SVM", "RMS"], key="adm_f_metode")
            with f3:
                f_q = st.text_input("Cari file / sesi ID", placeholder="Ketik kata kunci...", key="adm_f_q")

            data_riwayat = ambil_riwayat_terfilter(limit=100, vonis=f_vonis, metode=f_metode, query=f_q)
            if data_riwayat:
                df_admin = pd.DataFrame(data_riwayat)
                st.dataframe(df_admin, width="stretch")
                csv_data = df_admin.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Unduh Hasil Filter (CSV)",
                    data=csv_data,
                    file_name=f"log_kebisingan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
            else:
                st.info("Tidak ada data riwayat yang cocok dengan filter.")

        with sub_sesi:
            st.markdown("#### Sesi Klien Streamlit")
            kol_s1, kol_s2 = st.columns([4, 1])
            with kol_s2:
                if st.button("🧹 Bersihkan Sesi Mati", key="btn_bersihkan_sesi_st"):
                    bersihkan_sesi_manual()
                    st.success("Sesi tidak aktif dibersihkan.")
                    st.rerun()

            sesi_list = ambil_daftar_sesi()
            if sesi_list:
                st.dataframe(pd.DataFrame(sesi_list), width="stretch")
            else:
                st.caption("Belum ada sesi tercatat.")

        with sub_sistem:
            sis = ambil_status_sistem()
            st.markdown("#### Informasi Kesehatan Database & Model")
            st.json(sis)
            st.markdown("""
            ##### Rujukan Standar Kebisingan Ruangan:
            - **Zona Sunyi (< -30 dBFS / < 55 dBA)**: Perpustakaan, ruang rapat, ruang konsentrasi.
            - **Zona Waspada (-30 s.d. -20 dBFS / 55 - 70 dBA)**: Percakapan aktif, aktivitas normal.
            - **Zona Berisik (> -20 dBFS / > 85 dBA)**: Kegaduhan tinggi, peringatan diaktifkan.
            """)
