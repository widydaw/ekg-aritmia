import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.monitor import ecg_monitor

ui.page_setup("Deteksi R")
rec = ui.sidebar_record()
if rec is None:
    st.stop()
t0, t1 = ui.window_controls(rec)

ui.eyebrow("RM 2 · Deteksi puncak R")
st.title("Pan-Tompkins dan detak jantung")
st.write("Detektor buatan sendiri dijalankan pada sinyal bersih (band-pass 0,5–40 Hz). Hasilnya dibandingkan "
         "dengan anotasi dalam toleransi 150 ms.")

det = ui.cached_detection(rec.uid, rec)
peaks, clean, hr = det["peaks"], det["clean"], det["hr"]
c = st.columns(6)
c[0].metric("Detak terdeteksi", f"{len(peaks):,}")
c[1].metric("HR rata-rata", f"{hr['hr']:.0f} bpm")
if det["match"] is not None:
    pairs, tp, fp, fn = det["match"]
    c[2].metric("Sensitivitas", f"{tp / (tp + fn):.2%}", help=f"{fn} detak terlewat")
    c[3].metric("Presisi (PPV)", f"{tp / (tp + fp):.2%}", help=f"{fp} salah deteksi")
c[4].metric("SDNN", f"{hr['sdnn']:.0f} ms")
c[5].metric("RMSSD", f"{hr['rmssd']:.0f} ms")

st.subheader("Monitor berjalan")
st.caption("Sinyal bersih diputar mulai dari posisi 'Mulai' di panel kiri, 2 menit, berulang. Titik merah = puncak R hasil Pan-Tompkins.")
ecg_monitor(clean, rec.fs, peaks, None, start=t0, duration=120, title=rec.name)

st.subheader("Sinyal bersih dan puncak R")
fig = ui.add_trace(ui.ecg_figure(t0, t1, 320), clean, rec.fs, t0, t1)
win = peaks[(peaks >= t0 * rec.fs) & (peaks <= t1 * rec.fs)]
fig.add_trace(go.Scatter(x=win / rec.fs, y=clean[win], mode="markers", name="terdeteksi",
                         marker=dict(color=ui.ACCENT, size=8), hovertemplate="R %{x:.3f} s<extra></extra>"))
if det["match"] is not None:
    pairs = det["match"][0]
    md, mr = {i for i, _ in pairs}, {j for _, j in pairs}
    ref = rec.r_peaks
    fn_idx = np.array([r for j, r in enumerate(ref) if j not in mr and t0 * rec.fs <= r <= t1 * rec.fs], int)
    fp_idx = np.array([p for i, p in enumerate(peaks) if i not in md and t0 * rec.fs <= p <= t1 * rec.fs], int)
    if len(fn_idx):
        fig.add_trace(go.Scatter(x=fn_idx / rec.fs, y=clean[fn_idx], mode="markers", name="terlewat (FN)",
                                 marker=dict(symbol="circle-open", color="#a86400", size=16, line=dict(width=2))))
    if len(fp_idx):
        fig.add_trace(go.Scatter(x=fp_idx / rec.fs, y=clean[fp_idx], mode="markers", name="salah deteksi (FP)",
                                 marker=dict(symbol="x", color="#a86400", size=12)))
    ann = ref[(ref >= t0 * rec.fs) & (ref <= t1 * rec.fs)]
    ytop = clean[int(t0 * rec.fs):int(t1 * rec.fs)].max() * 1.12
    fig.add_trace(go.Scatter(x=ann / rec.fs, y=np.full(len(ann), ytop), mode="markers", name="anotasi",
                             marker=dict(symbol="line-ns-open", color=ui.INK3, size=10, line=dict(width=2))))
fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig, width="stretch")

st.subheader("Tahapan Pan-Tompkins")
pt = det["pt"]
stages = {"a · Band-pass 5–15 Hz": pt.bandpassed, "b · Turunan": pt.derivative,
          "c · Kuadrat": pt.squared, "d–e · Integrasi 150 ms + ambang": pt.integrated}
stage = st.radio("Tahap", list(stages), index=3, horizontal=True, label_visibility="collapsed")
t, y = ui.window_slice(stages[stage], rec.fs, t0, t1)
fig2 = go.Figure(go.Scatter(x=t, y=y, mode="lines", line=dict(color=ui.ACCENT if stage.startswith("d") else ui.INK, width=1.3), name=stage))
if stage.startswith("d"):
    ti, tv = pt.threshold_idx / rec.fs, pt.threshold_val
    m = (ti >= t0 - 2) & (ti <= t1)
    fig2.add_trace(go.Scatter(x=ti[m], y=tv[m], mode="lines", line=dict(color=ui.INK2, dash="dash", shape="hv"), name="ambang adaptif"))
for p in win:
    fig2.add_vline(x=p / rec.fs, line=dict(color=ui.ACCENT, width=1), opacity=0.25)
fig2.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor=ui.PAPER, paper_bgcolor="rgba(0,0,0,0)",
                   showlegend=False, xaxis=dict(range=[t0, t1], ticksuffix=" s", gridcolor="#ecdfe1"), yaxis=dict(gridcolor="#ecdfe1"))
st.plotly_chart(fig2, width="stretch")
st.caption({
    "a": "Band-pass 5–15 Hz menonjolkan energi QRS dan menekan P, T, serta baseline.",
    "b": "Turunan lima titik: y[n] = (2x[n+1] + x[n+2] − x[n−2] − 2x[n−1]) · fs/8.",
    "c": "Pengkuadratan membuat semua nilai positif dan memperbesar kemiringan curam.",
    "d": "Integrasi 150 ms. Garis putus-putus: ambang NPKI + 0,25 (SPKI − NPKI), dengan refrakter 200 ms, cek gelombang T, dan pencarian mundur.",
}[stage[0]])

st.subheader("Interval RR sepanjang rekaman")
rr = hr["rr_ms"]
fig3 = go.Figure(go.Scatter(x=peaks[1:] / rec.fs, y=rr, mode="lines+markers", line=dict(color=ui.ACCENT, width=1.2),
                            marker=dict(size=4), hovertemplate="%{x:.1f} s · RR %{y:.0f} ms<extra></extra>"))
fig3.add_vrect(x0=t0, x1=t1, fillcolor=ui.ACCENT, opacity=0.08, line_width=0)
fig3.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor=ui.PAPER, paper_bgcolor="rgba(0,0,0,0)",
                   xaxis=dict(ticksuffix=" s", gridcolor="#ecdfe1"), yaxis=dict(title="RR (ms)", gridcolor="#ecdfe1"))
st.plotly_chart(fig3, width="stretch")
