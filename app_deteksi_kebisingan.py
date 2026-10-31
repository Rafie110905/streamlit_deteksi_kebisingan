import io
import hashlib
import os
import threading
import wave
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

st.set_page_config(page_title="Deteksi Kebisingan Ruangan", layout="wide")

FRAME_LEN, HOP_LEN, SEED = 2048, 512, 42
FITUR_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fitur_audio_kebisingan.csv")
WARNA = {"berisik": "#f43f5e", "tenang": "#10b981", "ragu": "#f59e0b",
         "gelap": "#0f172a", "panel": "#1e293b", "teks": "#e2e8f0", "redup": "#94a3b8"}


def buat_audio_peringatan():
    sample_rate = 22050
    potongan = []
    for frekuensi, durasi in ((880, 0.18), (0, 0.10), (1046, 0.24)):
        jumlah = int(sample_rate * durasi)
        if frekuensi == 0:
            potongan.append(np.zeros(jumlah, dtype=np.float32))
            continue
        waktu = np.arange(jumlah) / sample_rate
        amplop = np.minimum(waktu / 0.015, 1) * np.minimum((durasi - waktu) / 0.02, 1)
        potongan.append(0.35 * np.sin(2 * np.pi * frekuensi * waktu) * amplop)

    pcm = (np.concatenate(potongan) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as file_audio:
        file_audio.setnchannels(1)
        file_audio.setsampwidth(2)
        file_audio.setframerate(sample_rate)
        file_audio.writeframes(pcm.tobytes())
    return buffer.getvalue()


PERINGATAN_AUDIO = buat_audio_peringatan()


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
                data = data.mean(axis=0)
            data = np.nan_to_num(data, copy=False)
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

                if (self._jumlah_sampel < ukuran_jendela
                        or self._sampel_sejak_prediksi < sr):
                    return
                y = np.concatenate(tuple(self._potongan))
                self._sampel_sejak_prediksi = 0

            f, _, _, _ = ekstrak_fitur_sinyal(y, sr)
            hasil = {"fitur": f}
            if model is not None and kolom is not None:
                x = pd.DataFrame([f])[kolom].values.astype(np.float32)
                idx1 = list(model.classes_).index(1)
                p = float(model.predict_proba(x)[0][idx1])
                suara = np.array([pohon.predict_proba(x)[0][idx1] for pohon in model.estimators_])
                hasil.update({
                    "p": p,
                    "n_berisik": int((suara >= .5).sum()),
                    "n_pohon": len(suara),
                })
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


def tentukan_vonis_random_forest(p):
    return p >= 0.5


def tentukan_vonis_rms(rms_dbfs, ambang_rms_dbfs):
    return rms_dbfs >= ambang_rms_dbfs


def putar_peringatan_rekaman(audio_id, metode, berisik):
    if not berisik:
        return
    peringatan_diputar = st.session_state.setdefault("peringatan_rekaman_diputar", set())
    kunci_peringatan = (audio_id, metode)
    if kunci_peringatan not in peringatan_diputar:
        st.warning("Peringatan: hasil deteksi BERISIK.")
        st.audio(PERINGATAN_AUDIO, format="audio/wav", autoplay=True)
        peringatan_diputar.add(kunci_peringatan)


def kartu_hasil_random_forest(nama, p, n_berisik, n_pohon, f, berisik):
    ragu = abs(p - .5) < .10
    warna = WARNA["ragu"] if ragu else WARNA["berisik"] if berisik else WARNA["tenang"]
    teks = "BERISIK" if berisik else "TIDAK BERISIK"
    kotak = lambda k, v, s="": (f"<div style='background:{WARNA['panel']};padding:8px 10px;border-radius:8px'>"
                                 f"<div style='color:{WARNA['redup']};font-size:10px'>{k}</div><b>{v}</b>"
                                 f"<div style='color:{WARNA['redup']};font-size:10px'>{s}</div></div>")
    return f"""
    <div style="background:{WARNA['gelap']};color:{WARNA['teks']};padding:20px 24px;border-radius:14px;
                border-left:8px solid {warna};border:1px solid #334155;border-left:8px solid {warna};font-family:'Segoe UI',Arial,sans-serif">
      <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:12px">
        <div>
          <div style="font-size:11px;letter-spacing:2px;color:{WARNA['redup']}">RANDOM FOREST · {nama}</div>
          <div style="font-size:32px;font-weight:800;color:{warna};margin-top:4px">{teks}</div>
          <div style="font-size:13px;color:{WARNA['redup']}">Keyakinan model: <b style="color:{WARNA['teks']}">{tingkat_keyakinan(p)}</b></div>
        </div>
        <div style="text-align:right">
          <div style="font-size:11px;color:{WARNA['redup']}">SUARA POHON (MODEL)</div>
          <div style="font-size:32px;font-weight:800">{n_berisik}<span style="font-size:15px;color:{WARNA['redup']}"> / {n_pohon}</span></div>
          <div style="font-size:11px;color:{WARNA['redup']}">memilih BERISIK</div>
        </div>
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

    ax1 = fig.add_subplot(gs[0, 0]); gaya(ax1, f"Voting {n_pohon} pohon")
    jml = [n_pohon - n_berisik, n_berisik]
    bars = ax1.bar(["Tidak Berisik", "Berisik"], jml, color=[WARNA["tenang"], WARNA["berisik"]], width=.55)
    for b, j in zip(bars, jml):
        ax1.text(b.get_x() + b.get_width() / 2, j + n_pohon * .015, f"{j} ({j / n_pohon * 100:.0f}%)",
                 ha="center", color=WARNA["teks"], fontsize=11, fontweight="bold")
    ax1.axhline(n_pohon / 2, color="white", ls="--", lw=1, alpha=.7)
    ax1.set_ylim(0, n_pohon * 1.12); ax1.tick_params(axis="x", labelcolor=WARNA["teks"], labelsize=10)

    ax2 = fig.add_subplot(gs[0, 1]); gaya(ax2, "Profil fitur penting (z-score vs dataset)")
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
    ["Random Forest", "RMS"],
    horizontal=True,
    key="metode_deteksi",
)

if metode_deteksi == "Random Forest":
    model, kolom, imp, ref, akurasi_cv, n_data = siapkan_model()
else:
    model, kolom, imp, ref = None, None, None, None

with st.sidebar:
    if metode_deteksi == "Random Forest":
        st.header("Tentang Model")
        st.metric("Akurasi validasi silang", f"{akurasi_cv * 100:.1f}%")
        st.caption(f"Random Forest · 300 pohon · {len(kolom)} fitur · {n_data} data latih. "
                   "Validasi 5-fold dikelompokkan per mahasiswa.")
        st.markdown("**Fitur terpenting**")
        st.bar_chart(imp.head(8)[::-1])
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
if "metode_deteksi_aktif" not in st.session_state:
    st.session_state.metode_deteksi_aktif = metode_deteksi
elif st.session_state.metode_deteksi_aktif != metode_deteksi:
    st.session_state.audio_langsung.reset()
    st.session_state.hasil_audio = None
    st.session_state.peringatan_realtime_aktif = False
    st.session_state.metode_deteksi_aktif = metode_deteksi

tab_uji, tab_riw = st.tabs(["Uji Sampel", "Riwayat"])

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
        st.info("Mode Random Forest menggunakan model saja; ambang RMS tidak memengaruhi vonis.")
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
        from streamlit_webrtc import webrtc_streamer

        st.caption("Tekan START dan izinkan mikrofon. Hasil diperbarui setiap detik dari 2 detik suara terakhir.")
        st.session_state.hasil_audio = None
        state_audio = st.session_state.audio_langsung

        def callback_audio(frame):
            if metode_deteksi == "Random Forest":
                state_audio.proses_frame(frame, model, kolom)
            else:
                state_audio.proses_frame(frame, None, None)
            return frame

        webrtc_ctx = webrtc_streamer(
            key=f"deteksi-kebisingan-realtime-{metode_deteksi}",
            audio_frame_callback=callback_audio,
            media_stream_constraints={"audio": True, "video": False},
            sendback_audio=False,
        )

        @st.fragment(run_every="1s")
        def tampilkan_hasil_realtime():
            hasil_live, error_live, nomor_hasil, jumlah_frame = state_audio.baca()
            if not webrtc_ctx.state.playing:
                st.session_state.peringatan_realtime_aktif = False
                st.info("Tekan START untuk mulai mendeteksi melalui mikrofon. Tekan STOP untuk mengakhiri.")
                return
            if error_live:
                st.error(f"Gagal menganalisis audio langsung: {error_live}")
            if hasil_live is None:
                if jumlah_frame:
                    st.info("Audio diterima. Menunggu 2 detik suara untuk prediksi pertama…")
                else:
                    st.warning("Belum menerima audio. Periksa izin mikrofon dan perangkat input di browser.")
                return

            fitur_live = hasil_live["fitur"]
            if metode_deteksi == "Random Forest":
                p_live = hasil_live["p"]
                berisik_live = tentukan_vonis_random_forest(p_live)
                st.markdown(kartu_hasil_random_forest(
                    "Deteksi realtime", p_live, hasil_live["n_berisik"],
                    hasil_live["n_pohon"], fitur_live, berisik_live,
                ), unsafe_allow_html=True)
                st.caption(
                    f"RMS: {fitur_live['rms_dbfs']:.1f} dBFS · "
                    f"P(berisik): {p_live:.3f} · Pembaruan ke-{nomor_hasil}"
                )
            else:
                berisik_live = tentukan_vonis_rms(fitur_live["rms_dbfs"], ambang_rms_dbfs)
                st.markdown(kartu_hasil_rms(
                    "Deteksi realtime", fitur_live, berisik_live, ambang_rms_dbfs
                ), unsafe_allow_html=True)
                st.caption(f"Pembaruan ke-{nomor_hasil} · RMS tidak menggunakan model Random Forest.")

            if berisik_live and not st.session_state.peringatan_realtime_aktif:
                st.warning("Peringatan: hasil deteksi realtime BERISIK.")
                st.audio(PERINGATAN_AUDIO, format="audio/wav", autoplay=True)
                st.session_state.peringatan_realtime_aktif = True
            elif not berisik_live:
                st.session_state.peringatan_realtime_aktif = False

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
            if metode_deteksi == "Random Forest":
                if "p" not in hasil:
                    x = pd.DataFrame([f])[kolom].values.astype(np.float32)
                    idx1 = list(model.classes_).index(1)
                    p = float(model.predict_proba(x)[0][idx1])
                    suara = np.array([t.predict_proba(x)[0][idx1] for t in model.estimators_])
                    hasil.update({
                        "p": p,
                        "n_berisik": int((suara >= .5).sum()),
                        "n_pohon": len(suara),
                    })
                p, n_b, n_p = hasil["p"], hasil["n_berisik"], hasil["n_pohon"]
                berisik = tentukan_vonis_random_forest(p)
                st.markdown(kartu_hasil_random_forest(
                    nama_audio, p, n_b, n_p, f, berisik
                ), unsafe_allow_html=True)
                if sumber_audio == "Rekam langsung":
                    putar_peringatan_rekaman(audio_id, metode_deteksi, berisik)
                st.write("")
                fig = buat_dashboard(f, y, sr, d, p, berisik, n_b, n_p, kolom, imp, ref)
                st.pyplot(fig)
                plt.close(fig)
                entri = {
                    "_audio_id": f"{audio_id}:Random Forest", "file": nama_audio,
                    "metode": "Random Forest",
                    "vonis": "BERISIK" if berisik else "TIDAK BERISIK",
                    "P model (berisik)": round(p, 3), "pohon_berisik": f"{n_b}/{n_p}",
                    "keyakinan model": tingkat_keyakinan(p),
                }
            else:
                berisik = tentukan_vonis_rms(f["rms_dbfs"], ambang_rms_dbfs)
                st.markdown(kartu_hasil_rms(
                    nama_audio, f, berisik, ambang_rms_dbfs
                ), unsafe_allow_html=True)
                if sumber_audio == "Rekam langsung":
                    putar_peringatan_rekaman(audio_id, metode_deteksi, berisik)
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
