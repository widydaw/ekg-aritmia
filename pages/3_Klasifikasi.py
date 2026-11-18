import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.monitor import ecg_monitor
from src.classifier import MODEL_NAMES
from src.data_loader import CLASSES
from src.features import segments

ui.page_setup("Klasifikasi")
rec = ui.sidebar_record()
if rec is None:
    st.stop()
t0, t1 = ui.window_controls(rec)

ui.eyebrow("RM 5 · Alur menyeluruh")
st.title("Hasil klasifikasi per detak")
st.write("Setiap detak dari detektor dipotong 0,2 s sebelum dan 0,4 s sesudah puncak R, diringkas menjadi 38 fitur, "
         "lalu diklasifikasikan oleh model yang dilatih pada DS1.")

models = ui.load_models()
if not models:
    ui.models_missing_message()
    st.stop()
key = st.radio("Model", list(models), format_func=MODEL_NAMES.get, horizontal=True)
model, meta = models[key]
st.caption(f"Model dilatih pada data {meta.get('source', '?')} ({meta.get('created', '?')}), "
           f"F1 makro DS2 = {meta.get('macro_f1_ds2', float('nan')):.3f}.")
if rec.fs != meta.get("fs", 360):
    st.warning(f"Rekaman ini {rec.fs:g} Hz, sedangkan model dilatih pada {meta.get('fs')} Hz.")

det = ui.cached_detection(rec.uid, rec)
res = ui.cached_classification(rec.uid, rec, key, model)
peaks, clean, pred, truth = det["peaks"], det["clean"], res["pred"], res["truth"]
n = len(pred)

c = st.columns(4)
for col, k in zip(c, CLASSES):
    cnt = int(np.sum(pred == k))
    col.metric(ui.class_label(k), f"{cnt:,} detak", f"{cnt / max(n, 1):.1%} dari total", delta_color="off")
if truth is not None:
    m = truth != ""
    m &= np.isin(truth, CLASSES)
    if m.any():
        c[3].metric("Kesesuaian dg. anotasi", f"{np.mean(pred[m] == truth[m]):.1%}", help="Pada detak yang cocok dengan anotasi")

st.subheader("Monitor berjalan")
st.caption("Diputar mulai dari posisi 'Mulai' di panel kiri, 2 menit, berulang. Huruf di atas tiap detak = kelas prediksi.")
ecg_monitor(clean, rec.fs, peaks, pred, start=t0, duration=120, title=f"{rec.name} · {MODEL_NAMES[key]}")

st.subheader("Detak berwarna menurut kelas")
fig = ui.add_trace(ui.ecg_figure(t0, t1, 330), clean, rec.fs, t0, t1)
inwin = (peaks >= t0 * rec.fs) & (peaks <= t1 * rec.fs)
seg = clean[int(t0 * rec.fs):int(t1 * rec.fs) + 1]
ytop = seg.max() + 0.15 * (seg.max() - seg.min())
for k in CLASSES:
    sel = inwin & (pred == k)
    if not sel.any():
        continue
    pk = peaks[sel]
    fig.add_trace(go.Scatter(x=pk / rec.fs, y=clean[pk], mode="markers", marker=dict(color=ui.CLASS_COLORS[k], size=9),
                             name=k, hovertemplate=f"{k} · %{{x:.2f}} s<extra></extra>"))
    fig.add_trace(go.Scatter(x=pk / rec.fs, y=np.full(len(pk), ytop), mode="text", text=[k] * len(pk),
                             textfont=dict(color=ui.CLASS_COLORS[k], size=13, family="IBM Plex Mono, monospace"),
                             hoverinfo="skip"))
    if k != "N":
        for p in pk:
            fig.add_vrect(x0=p / rec.fs - 0.12, x1=p / rec.fs + 0.22, fillcolor=ui.CLASS_COLORS[k], opacity=0.12, line_width=0)
if truth is not None:
    diff = inwin & (truth != "") & (truth != pred)
    if diff.any():
        pk = peaks[diff]
        fig.add_trace(go.Scatter(x=pk / rec.fs, y=np.full(len(pk), seg.min() - 0.1 * (seg.max() - seg.min())), mode="text",
                                 text=[f"acuan {t}" for t in truth[diff]], textfont=dict(color=ui.INK3, size=11), hoverinfo="skip"))
st.plotly_chart(fig, width="stretch")

st.subheader("Bentuk detak per kelas")
cols = st.columns(3)
B = int(round(0.2 * rec.fs))
for col, k in zip(cols, CLASSES):
    ids = np.flatnonzero(pred == k)
    with col:
        st.markdown(f"**{ui.class_label(k)}** · {len(ids)} detak")
        figk = go.Figure()
        if len(ids):
            pick = ids[:: max(1, len(ids) // 60)]
            S = segments(clean, peaks[pick], rec.fs)
            S = S - S[:, :20].mean(axis=1, keepdims=True)
            tx = (np.arange(S.shape[1]) - B) / rec.fs
            for row in S:
                figk.add_trace(go.Scatter(x=tx, y=row, mode="lines", line=dict(color=ui.CLASS_COLORS[k], width=1), opacity=0.25, hoverinfo="skip"))
            figk.add_trace(go.Scatter(x=tx, y=S.mean(axis=0), mode="lines", line=dict(color=ui.INK, width=2.2), name="rata-rata"))
        figk.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, plot_bgcolor=ui.PAPER,
                           paper_bgcolor="rgba(0,0,0,0)", xaxis=dict(range=[-0.2, 0.4], ticksuffix=" s", gridcolor="#ecdfe1"),
                           yaxis=dict(gridcolor="#ecdfe1"))
        st.plotly_chart(figk, width="stretch", config={"displayModeBar": False})

st.subheader("Tabel detak")
F = res["features"]
df = pd.DataFrame({
    "No": np.arange(1, n + 1),
    "Waktu (s)": peaks / rec.fs,
    "RR sebelum (ms)": F[:, 0] * 1000,
    "RR sesudah (ms)": F[:, 1] * 1000,
    "Lebar QRS (ms)": F[:, 7],
    "Prediksi": pred,
})
if res["prob"] is not None:
    df["Keyakinan"] = res["prob"].max(axis=1)
if truth is not None:
    df["Label acuan"] = truth
flt = st.segmented_control("Tampilkan", ["Semua", "N", "S", "V"] + (["Berbeda dari acuan"] if truth is not None else []), default="Semua")
view = df
if flt in CLASSES:
    view = df[df["Prediksi"] == flt]
elif flt == "Berbeda dari acuan":
    view = df[(df["Label acuan"] != "") & (df["Label acuan"] != df["Prediksi"])]
st.dataframe(view, hide_index=True, width="stretch", height=380,
             column_config={"Waktu (s)": st.column_config.NumberColumn(format="%.2f"),
                            "RR sebelum (ms)": st.column_config.NumberColumn(format="%.0f"),
                            "RR sesudah (ms)": st.column_config.NumberColumn(format="%.0f"),
                            "Lebar QRS (ms)": st.column_config.NumberColumn(format="%.0f"),
                            "Keyakinan": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")})
st.download_button("Unduh hasil (CSV)", df.to_csv(index=False).encode("utf-8"),
                   file_name=f"hasil_{rec.uid}_{key}.csv", mime="text/csv")
