"""Komponen bersama aplikasi Streamlit: pemilih rekaman, cache, jendela waktu, dan grafik.

Halaman web hanya memanggil fungsi di src/, sehingga angka di web sama dengan eksperimen.
"""
from __future__ import annotations

import json

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from .classifier import MODEL_DIR, MODEL_NAMES, load_model, predict, predict_proba
from .data_loader import CLASSES, DS1, DS2, Record, load_csv, load_record
from .features import extract_features
from .preprocessing import STANDARD, preprocess
from .qrs_detection import heart_rate, match_beats, pan_tompkins
from .synthetic import SAMPLES, sample_record

# Palet "kertas EKG"
INK, INK2, INK3 = "#1b2130", "#4b5366", "#868c9c"
ACCENT = "#b3243f"
RAW = "#9aa1b2"
PAPER = "#fffdfc"
GRID_MAJOR, GRID_MINOR = "rgba(214,84,108,0.32)", "rgba(214,84,108,0.12)"
CLASS_COLORS = {"N": "#2b7a76", "S": "#c17a07", "V": "#8d2e86"}

SRC_MIT, SRC_SYN, SRC_CSV = "MIT-BIH (PhysioNet)", "Contoh sintetis", "Unggah CSV"

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
def _keep(key: str, default):
    """Widget Streamlit dihapus saat pindah halaman; simpan salinannya di session_state."""
    store = f"_{key}"
    if store not in st.session_state:
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


@st.cache_data(show_spinner=False, max_entries=4)
def _load_csv(data: bytes, fs: float, filename: str) -> Record:
    return load_csv(data, fs, name=filename)


def sidebar_record() -> Record | None:
    sb = st.sidebar
    sb.header("Rekaman")
    _keep("src", SRC_SYN)
    src = sb.radio("Sumber", [SRC_MIT, SRC_SYN, SRC_CSV], key="src", on_change=_save, args=("src",))
    rec = None
    if src == SRC_MIT:
        _keep("mit", DS2[0])
        name = sb.selectbox("Nomor rekaman", DS2 + DS1, key="mit", on_change=_save, args=("mit",),
                            format_func=lambda r: f"{r} · {'uji (DS2)' if r in DS2 else 'latih (DS1)'}")
        try:
            rec = _load_mit(name)
        except Exception as e:  # tanpa internet atau berkas rusak
            sb.error(f"Rekaman {name} gagal dimuat: {e}. Periksa koneksi internet, atau jalankan "
                     "`python -c \"from src.data_loader import download_mitdb; download_mitdb()\"` sekali.")
            return None
    elif src == SRC_SYN:
        _keep("syn", "SIN-02")
        sid = sb.selectbox("Contoh", list(SAMPLES), key="syn", on_change=_save, args=("syn",),
                           format_func=lambda s: f"{s} · {SAMPLES[s][0]}")
        rec = _load_syn(sid)
    else:
        _keep("csv_fs", 360)
        fs = sb.number_input("Frekuensi sampling (Hz)", 50, 2000, step=1, key="csv_fs",
                             on_change=_save, args=("csv_fs",))
        up = sb.file_uploader("Berkas CSV satu kolom", type=["csv", "txt"],
                              help="Satu nilai sinyal (mV) per baris, boleh dengan satu baris judul. 10 detik sampai 30 menit.")
        if up is not None:
            st.session_state["upload"] = (up.getvalue(), up.name)
        if "upload" not in st.session_state:
            sb.info("Unggah berkas untuk mulai. Isi frekuensi sampling lebih dulu.")
            return None
        data, fname = st.session_state["upload"]
        try:
            rec = _load_csv(data, float(fs), fname)
        except ValueError as e:
            sb.error(str(e))
            return None
    sb.caption(rec.description)
    return rec


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


# ---------- analisis standar (sama dengan eksperimen) ----------
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
def load_models() -> dict:
    out = {}
    for k in MODEL_NAMES:
        m, meta = load_model(k)
        if m is not None:
            out[k] = (m, meta)
    return out


def load_results() -> dict | None:
    p = MODEL_DIR / "results.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def models_missing_message() -> None:
    st.warning("Model belum ada. Jalankan `python train.py` (MIT-BIH) atau `python train.py --synthetic` "
               "(cepat, tanpa internet) dari folder proyek, lalu muat ulang halaman ini.")


@st.cache_data(show_spinner="Mengklasifikasikan detak…", max_entries=16)
def cached_classification(uid: str, _rec: Record, model_key: str, _model) -> dict:
    det = cached_detection(uid, _rec)
    F = extract_features(det["clean"], det["peaks"], _rec.fs)
    pred = predict(_model, F)
    prob = predict_proba(_model, F)
    truth = None
    if det["match"] is not None:
        truth = np.full(len(det["peaks"]), "", dtype=object)
        for i, j in det["match"][0]:
            truth[i] = _rec.labels[j]
    return {"features": F, "pred": pred, "prob": prob, "truth": truth}


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


def class_label(c: str) -> str:
    return {"N": "N · Normal", "S": "S · Supraventrikular prematur", "V": "V · Ventrikular prematur"}.get(c, c)


__all__ = ["CLASSES", "CLASS_COLORS"]
