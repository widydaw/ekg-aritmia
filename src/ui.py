"""Komponen bersama aplikasi Streamlit: pemilih rekaman, cache, jendela waktu, dan grafik.

Halaman web hanya memanggil fungsi di src/, sehingga angka di web sama dengan eksperimen.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from .data_loader import DS1, DS2, Record, load_record
from .file_formats import EXTENSIONS, FORMATS, default_channel, read_upload, to_record
from .preprocessing import STANDARD, preprocess
from .qrs_detection import heart_rate, match_beats, pan_tompkins
from .synthetic import SAMPLES, sample_record

# Palet "kertas EKG"
INK, INK2, INK3 = "#1b2130", "#4b5366", "#868c9c"
ACCENT = "#b3243f"
RAW = "#9aa1b2"
PAPER = "#fffdfc"
GRID_MAJOR, GRID_MINOR = "rgba(214,84,108,0.32)", "rgba(214,84,108,0.12)"

SRC_MIT, SRC_SYN, SRC_UP = "MIT-BIH (PhysioNet)", "Contoh sintetis", "Unggah berkas"

CSS = """
<style>
h1, h2, h3 { letter-spacing: -0.015em; }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
.eyebrow { font-family: ui-monospace, Consolas, monospace; font-size: .74rem; letter-spacing: .12em;
           text-transform: uppercase; color: #b3243f; margin-bottom: -.4rem; }
.disclaimer { color: #a86400; font-size: .85rem; }
</style>
"""


def page_setup(title: str) -> None:
    st.set_page_config(page_title=f"{title} · EKG Aritmia", page_icon="🫀", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.markdown('<span class="disclaimer">Untuk pembelajaran, bukan alat diagnosis medis.</span>',
                        unsafe_allow_html=True)


def eyebrow(text: str) -> None:
    st.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)


# ---------- nilai widget yang bertahan antarhalaman ----------
def _keep(key: str, default, allowed=None):
    """Widget Streamlit dihapus saat pindah halaman; simpan salinannya di session_state."""
    store = f"_{key}"
    if store not in st.session_state or (allowed is not None and st.session_state[store] not in allowed):
        st.session_state[store] = default
    st.session_state[key] = st.session_state[store]


def _save(key: str):
    st.session_state[f"_{key}"] = st.session_state[key]


# ---------- pemuatan rekaman (di-cache) ----------
@st.cache_data(show_spinner="Memuat rekaman MIT-BIH dari PhysioNet…", max_entries=8)
def _load_mit(name: int) -> Record:
    return load_record(name)


@st.cache_data(show_spinner=False, max_entries=8)
def _load_syn(sid: str) -> Record:
    return sample_record(sid)


@st.cache_data(show_spinner="Membaca berkas…", max_entries=4)
def _read(files: tuple, fs: float):
    return read_upload(list(files), fs)


def _upload_help() -> str:
    return "Format yang didukung:\n\n" + "\n".join(f"- **{k}**: {v}" for k, v in FORMATS.items())


def sidebar_record() -> Record | None:
    sb = st.sidebar
    sb.header("Rekaman")
    sources = [SRC_MIT, SRC_SYN, SRC_UP]
    _keep("src", SRC_SYN, sources)
    src = sb.radio("Sumber", sources, key="src", on_change=_save, args=("src",))
    rec = None
    if src == SRC_MIT:
        _keep("mit", 100)
        name = sb.selectbox("Nomor rekaman", sorted(DS1 + DS2), key="mit", on_change=_save, args=("mit",))
        try:
            rec = _load_mit(name)
        except Exception as e:  # tanpa internet atau berkas rusak
            sb.error(f"Rekaman {name} gagal dimuat: {e}. Periksa koneksi internet, lalu coba lagi.")
            return None
    elif src == SRC_SYN:
        _keep("syn", "SIN-02")
        sid = sb.selectbox("Contoh", list(SAMPLES), key="syn", on_change=_save, args=("syn",),
                           format_func=lambda s: f"{s} · {SAMPLES[s][0]}")
        rec = _load_syn(sid)
    else:
        rec = _upload_section(sb)
        if rec is None:
            return None
    sb.caption(rec.description)
    return rec


def _upload_section(sb) -> Record | None:
    ups = sb.file_uploader("Berkas EKG", type=EXTENSIONS, accept_multiple_files=True,
                           help=_upload_help())
    if ups:
        st.session_state["upload"] = tuple((u.name, u.getvalue()) for u in ups)
    _keep("up_fs", 360)
    fs = sb.number_input("Frekuensi sampling (Hz)", 50, 2000, step=1, key="up_fs", on_change=_save, args=("up_fs",),
                         help="Dipakai untuk CSV/TXT dan .mat yang tidak menyimpan frekuensi sampling. "
                              "WFDB, EDF, dan HL7 aECG membaca nilainya dari berkas.")
    with sb.expander("Format yang didukung"):
        st.markdown(_upload_help())
    if "upload" not in st.session_state:
        sb.info("Unggah berkas untuk mulai. Untuk WFDB, pilih .hea dan .dat (serta .atr bila ada) sekaligus.")
        return None
    files = st.session_state["upload"]
    try:
        up = _read(files, float(fs))
    except ValueError as e:
        sb.error(str(e))
        return None
    except Exception as e:  # berkas rusak atau format tak terduga
        sb.error(f"Berkas tidak bisa dibaca ({type(e).__name__}: {e}). Periksa formatnya.")
        return None
    names = list(up.channels)
    ch = default_channel(up)
    if len(names) > 1:
        _keep("up_ch", ch, names)
        ch = sb.selectbox("Sadapan", names, key="up_ch", on_change=_save, args=("up_ch",),
                          help="Rekaman berisi beberapa sadapan. Sadapan II (MLII) paling mirip dengan MIT-BIH.")
    try:
        return to_record(up, ch)
    except ValueError as e:
        sb.error(str(e))
        return None


def window_controls(rec: Record) -> tuple[float, float]:
    """Jendela waktu 5–30 s; rekaman 30 menit terlalu berat bila digambar sekaligus."""
    sb = st.sidebar
    sb.header("Jendela waktu")
    _keep("win", 10)
    width = sb.select_slider("Lebar (detik)", [5, 10, 20, 30], key="win", on_change=_save, args=("win",))
    width = min(width, rec.duration)
    max_start = max(0.0, rec.duration - width)
    if st.session_state.get("_start", 0.0) > max_start:
        st.session_state["_start"] = 0.0
    _keep("start", 0.0)
    if max_start > 0:
        start = sb.slider("Mulai (detik)", 0.0, float(round(max_start, 1)), step=0.5, key="start",
                          on_change=_save, args=("start",))
    else:
        start = 0.0
    return start, start + width


# ---------- analisis standar ----------
@st.cache_data(show_spinner="Menyaring sinyal…", max_entries=16)
def cached_preprocess(uid: str, _rec: Record, mode: str, low: float, high: float, order: int, notch):
    return preprocess(_rec.signal, _rec.fs, mode=mode, low=low, high=high, order=order, notch=notch)


@st.cache_data(show_spinner="Mendeteksi puncak R…", max_entries=8)
def cached_detection(uid: str, _rec: Record) -> dict:
    clean = preprocess(_rec.signal, _rec.fs, **STANDARD)
    pt = pan_tompkins(clean, _rec.fs, refine_on=clean)
    out = {"clean": clean, "pt": pt, "peaks": pt.peaks, "hr": heart_rate(pt.peaks, _rec.fs), "match": None}
    if _rec.has_annotations:
        out["match"] = match_beats(pt.peaks, _rec.r_peaks, _rec.fs)
    return out


@st.cache_resource(show_spinner="Memuat model…")
def cached_model(name: str):
    from .classifier import load_model
    return load_model(name)


@st.cache_data(show_spinner="Mengklasifikasi detak…", max_entries=8)
def cached_classification(uid: str, _rec: Record, model_name: str) -> dict | None:
    """Fitur dari puncak hasil Pan-Tompkins → kelas tiap detak (alur yang sama dengan evaluasi menyeluruh di train.py)."""
    from .classifier import predict, predict_proba
    from .features import extract_features

    model, meta = cached_model(model_name)
    if model is None:
        return None
    det = cached_detection(uid, _rec)
    F = extract_features(det["clean"], det["peaks"], _rec.fs)
    out = {"pred": predict(model, F), "proba": predict_proba(model, F), "classes": list(model.classes_), "meta": meta,
           "truth": None}
    if det["match"] is not None:
        pairs = det["match"][0]
        out["truth"] = [(i, _rec.labels[j]) for i, j in pairs]
    return out


# ---------- grafik ----------
def ecg_figure(t0: float, t1: float, height: int = 300) -> go.Figure:
    """Grafik bergaya kertas EKG: garis besar 1 s / 0,5 mV, garis kecil 0,2 s / 0,1 mV."""
    span = t1 - t0
    fig = go.Figure()
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor=PAPER,
                      paper_bgcolor="rgba(0,0,0,0)", showlegend=False, hovermode="x",
                      font=dict(family="IBM Plex Sans, Segoe UI, sans-serif", color=INK2, size=12))
    fig.update_xaxes(range=[t0, t1], dtick=1 if span <= 12 else 2, ticksuffix=" s", gridcolor=GRID_MAJOR,
                     minor=dict(dtick=0.2, gridcolor=GRID_MINOR, showgrid=span <= 20), zeroline=False,
                     showline=True, linecolor=GRID_MAJOR, mirror=True)
    fig.update_yaxes(dtick=0.5, ticksuffix=" mV", gridcolor=GRID_MAJOR, zeroline=False,
                     minor=dict(dtick=0.1, gridcolor=GRID_MINOR, showgrid=True),
                     showline=True, linecolor=GRID_MAJOR, mirror=True)
    return fig


def window_slice(x, fs: float, t0: float, t1: float):
    i0, i1 = int(max(0, t0 * fs)), int(min(len(x), t1 * fs + 1))
    return np.arange(i0, i1) / fs, np.asarray(x)[i0:i1]


def add_trace(fig, x, fs, t0, t1, color=INK, width=1.3, name=""):
    t, y = window_slice(x, fs, t0, t1)
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines", line=dict(color=color, width=width), name=name,
                             hovertemplate="%{x:.3f} s · %{y:.3f} mV<extra></extra>"))
    return fig
