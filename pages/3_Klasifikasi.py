import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.classifier import MODEL_NAMES
from src.data_loader import CLASS_NAMES, CLASSES
from src.features import segments
from src.monitor import CLASS_COLORS, ecg_monitor
from src.rhythm import AF_FRAC, BRADY, IRREG_THR, PERIOD_FRAC, PREMATURE_PCT, TACHY, WINDOW, summarize

ui.page_setup("Klasifikasi")
rec = ui.sidebar_record()
if rec is None:
    st.stop()
t0, t1 = ui.window_controls(rec)

st.sidebar.header("Model")
ui._keep("model", "rf", list(MODEL_NAMES))
model_name = st.sidebar.radio("Pengklasifikasi", list(MODEL_NAMES), key="model", on_change=ui._save, args=("model",),
                              format_func=MODEL_NAMES.get,
                              help="Dilatih pada DS1 MIT-BIH (22 pasien) dan diuji pada DS2 (22 pasien lain).")

ui.eyebrow("RM 3 · Klasifikasi detak")
st.title("Klasifikasi detak N, S, dan V")
st.write("Setiap detak hasil Pan-Tompkins diubah menjadi 38 fitur (interval RR, morfologi, wavelet db4, dan fitur "
         "relatif terhadap detak median pasien), lalu digolongkan oleh model menurut kelas AAMI: "
         "**N** normal, **S** supraventrikular prematur, **V** ventrikular prematur.")

res = ui.cached_classification(rec.uid, rec, model_name)
if res is None:
    st.error(f"Berkas model `models/{model_name}.joblib` tidak ditemukan. Jalankan `python train.py` terlebih dahulu.")
    st.stop()

det = ui.cached_detection(rec.uid, rec)
peaks, clean, pred = det["peaks"], det["clean"], res["pred"]
if len(peaks) < 3:
    st.warning("Terlalu sedikit detak terdeteksi untuk diklasifikasi.")
    st.stop()

rs = summarize(peaks, pred, rec.fs)
_box = {"warn": st.warning, "mild": st.info, "ok": st.success}[rs["verdict"][1]]
_icon = {"warn": "⚠️", "mild": "ℹ️", "ok": "✅", "na": "–"}
hr_txt = f"{rs['hr']:.0f} bpm, " if np.isfinite(rs["hr"]) else ""
_box(
    f"**Ringkasan irama: {rs['verdict'][0]}**\n\n"
    f"- {_icon[rs['premature'][1]]} **Detak prematur:** {rs['premature'][0]}\n"
    f"- {_icon[rs['rate'][1]]} **Kecepatan:** {hr_txt}{rs['rate'][0]}\n"
    f"- {_icon[rs['regularity'][1]]} **Keteraturan RR:** {rs['regularity'][0]}"
    + (f"\n\n{rs['note']}" if rs["note"] else "")
)
with st.expander("Aturan yang dipakai ringkasan ini"):
    st.markdown(
        f"- **Detak prematur** = jumlah detak S dan V hasil model. Disebut *sering* bila ≥ {PREMATURE_PCT:g}% dari "
        "semua detak; di bawah itu *sesekali*.\n"
        f"- **Kecepatan** = HR rata-rata (60 / rata-rata RR): < {BRADY} bpm bradikardia, > {TACHY} bpm takikardia.\n"
        f"- **Keteraturan**: rekaman dipotong per {WINDOW} interval RR. RR di sekitar detak V dibuang agar "
        "bigemini tidak terbaca sebagai fibrilasi. Satu potongan disebut tidak teratur bila median selisih RR "
        f"berturutan > {IRREG_THR:.0%} dari median RR. ≥ {AF_FRAC:.0%} potongan tidak teratur → curiga fibrilasi "
        f"atrium; {PERIOD_FRAC:.0%}–{AF_FRAC:.0%} → ada periode tidak teratur.\n\n"
        "Aturan ini sederhana dan hanya untuk pembelajaran, bukan diagnosis medis."
    )

n_cls = {c: int(np.sum(pred == c)) for c in CLASSES}
c = st.columns(4)
c[0].metric("Detak diklasifikasi", f"{len(pred):,}")
for col, k in zip(c[1:], CLASSES):
    col.metric(f"{k} · {CLASS_NAMES[k]}", f"{n_cls[k]:,}", help=f"{n_cls[k] / len(pred):.1%} dari semua detak")

if rec.lead != "?" and "II" not in rec.lead.upper().replace("III", ""):
    st.info(f"Model dilatih pada sadapan MLII. Sadapan rekaman ini **{rec.lead}**, jadi hasilnya bisa kurang tepat.")

st.subheader("Monitor berjalan dengan label kelas")
st.caption("Label N/S/V muncul di atas tiap detak saat tergambar. Warna mengikuti kelas hasil model.")
ecg_monitor(clean, rec.fs, peaks, pred, start=t0, duration=120, title=f"{rec.name} · {MODEL_NAMES[model_name]}")

st.subheader("Sinyal bersih dan kelas tiap detak")
fig = ui.add_trace(ui.ecg_figure(t0, t1, 320), clean, rec.fs, t0, t1)
m = (peaks >= t0 * rec.fs) & (peaks <= t1 * rec.fs)
for k in CLASSES:
    sel = peaks[m & (pred == k)]
    if len(sel):
        fig.add_trace(go.Scatter(x=sel / rec.fs, y=clean[sel], mode="markers+text", name=f"{k} · {CLASS_NAMES[k]}",
                                 text=[k] * len(sel), textposition="top center", textfont=dict(color=CLASS_COLORS[k], size=13),
                                 marker=dict(color=CLASS_COLORS[k], size=9), hovertemplate=f"{k} %{{x:.3f}} s<extra></extra>"))
fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig, width="stretch")

c1, c2 = st.columns(2, gap="large")
with c1:
    st.subheader("Bentuk rata-rata per kelas")
    seg = segments(clean, peaks, rec.fs)
    tt = (np.arange(seg.shape[1]) - int(round(0.2 * rec.fs))) / rec.fs * 1000
    fig2 = go.Figure()
    for k in CLASSES:
        sel = pred == k
        if sel.sum():
            fig2.add_trace(go.Scatter(x=tt, y=np.median(seg[sel], axis=0), mode="lines", name=f"{k} (n={int(sel.sum())})",
                                      line=dict(color=CLASS_COLORS[k], width=2)))
    fig2.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor=ui.PAPER, paper_bgcolor="rgba(0,0,0,0)",
                       legend=dict(orientation="h", y=1.12), xaxis=dict(title="ms dari puncak R", gridcolor="#ecdfe1"),
                       yaxis=dict(ticksuffix=" mV", gridcolor="#ecdfe1"))
    st.plotly_chart(fig2, width="stretch")
    st.caption("Median segmen −200 s.d. +400 ms. Detak V biasanya lebar dan tanpa gelombang P; detak S mirip N tetapi datang lebih awal.")
with c2:
    st.subheader("RR sebelum vs RR sesudah")
    rr = np.diff(peaks) / rec.fs
    pre, post = rr[:-1], rr[1:]
    lab = pred[1:-1]
    fig3 = go.Figure()
    for k in CLASSES:
        sel = lab == k
        if sel.sum():
            fig3.add_trace(go.Scattergl(x=pre[sel], y=post[sel], mode="markers", name=k,
                                        marker=dict(color=CLASS_COLORS[k], size=5, opacity=0.6)))
    fig3.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor=ui.PAPER, paper_bgcolor="rgba(0,0,0,0)",
                       legend=dict(orientation="h", y=1.12), xaxis=dict(title="RR sebelum (s)", gridcolor="#ecdfe1"),
                       yaxis=dict(title="RR sesudah (s)", gridcolor="#ecdfe1"))
    st.plotly_chart(fig3, width="stretch")
    st.caption("Detak prematur (S, V) berada di kiri atas: RR sebelum pendek, diikuti jeda kompensasi yang panjang.")

with st.expander("Unduh hasil per detak (CSV)"):
    df = pd.DataFrame({"waktu_s": peaks / rec.fs, "kelas": pred})
    if res["proba"] is not None:
        for j, k in enumerate(res["classes"]):
            df[f"prob_{k}"] = res["proba"][:, j]
    st.download_button("Unduh CSV", df.to_csv(index=False).encode(), file_name=f"klasifikasi_{rec.uid}_{model_name}.csv",
                       mime="text/csv")
    st.dataframe(df.head(200), hide_index=True, width="stretch")
