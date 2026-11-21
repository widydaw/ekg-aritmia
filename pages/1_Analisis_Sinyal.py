import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.preprocessing import MODES, filter_metrics, spectrum

ui.page_setup("Analisis Sinyal")
rec = ui.sidebar_record()
if rec is None:
    st.stop()
t0, t1 = ui.window_controls(rec)

ui.eyebrow("RM 1 · Praproses")
st.title("Analisis sinyal dan perancangan filter")
st.write("Rentang 0,5–40 Hz mempertahankan energi QRS, gelombang P, dan T, sambil menekan baseline wander, "
         "derau otot, dan jala-jala listrik. Filter Butterworth diterapkan dua arah (sosfiltfilt) agar fase tidak bergeser.")

with st.container(border=True):
    c = st.columns([2.2, 1, 1, 1, 1])
    mode = c[0].radio("Konfigurasi", list(MODES), format_func=MODES.get, horizontal=True)
    order = c[1].selectbox("Orde", [2, 4, 6], index=1)
    low = c[2].number_input("f rendah (Hz)", 0.05, 5.0, 0.5, 0.05, disabled=mode == "median")
    high = c[3].number_input("f tinggi (Hz)", 10.0, 150.0, 40.0, 1.0)
    notch = c[4].selectbox("Notch (Hz)", [50, 60], disabled=mode != "bandpass_notch")
    st.caption("Notch mengikuti frekuensi jaringan listrik setempat. **50 Hz**: Indonesia, sebagian besar Asia, Eropa, Afrika, "
               "Australia. **60 Hz**: Amerika Serikat, Kanada, sebagian Amerika Selatan (mis. Brasil), Korea Selatan, "
               "Taiwan, Filipina, Arab Saudi. Jepang memakai keduanya (timur 50 Hz, barat 60 Hz). "
               "MIT-BIH direkam di AS, jadi pakai 60 Hz; rekaman sendiri di Indonesia pakai 50 Hz.")
notch_val = float(notch) if mode == "bandpass_notch" else None
clean = ui.cached_preprocess(rec.uid, rec, mode, low, high, order, notch_val)


@st.cache_data(show_spinner=False, max_entries=32)
def metrics(uid, _rec, _clean, key):
    return filter_metrics(_rec.signal, _clean, _rec.fs, _rec.clean_reference, _rec.r_peaks)


m = metrics(rec.uid, rec, clean, (mode, low, high, order, notch_val))
cols = st.columns(5)
if "snr_after" in m:
    cols[0].metric("SNR thd. acuan", f"{m['snr_after']:.1f} dB", f"{m['snr_after'] - m['snr_before']:+.1f} dB")
cols[1].metric("Baseline < 0,5 Hz", f"−{m['baseline_db']:.1f} dB", help="Penurunan daya, sebelum dibanding sesudah")
cols[2].metric("Jala-jala 50/60 Hz", f"−{m['mains_db']:.1f} dB")
cols[3].metric("Derau > 40 Hz", f"−{m['hf_db']:.1f} dB")
if "qrs_corr" in m:
    cols[4].metric("Kemiripan QRS", f"{m['qrs_corr']:.3f}", help="Korelasi ±50 ms di sekitar R terhadap sinyal acuan bersih")

st.subheader("Sinyal asli")
st.plotly_chart(ui.add_trace(ui.ecg_figure(t0, t1, 260), rec.signal, rec.fs, t0, t1, color=ui.RAW, width=1.1),
                width="stretch")
st.subheader("Setelah penyaringan")
st.plotly_chart(ui.add_trace(ui.ecg_figure(t0, t1, 260), clean, rec.fs, t0, t1), width="stretch")

st.subheader("Spektrum daya (Welch)")
f, Pr = spectrum(rec.signal, rec.fs)
_, Pc = spectrum(clean, rec.fs)
sel = (f > 0) & (f <= min(100, rec.fs / 2))
fig = go.Figure()
fig.add_vrect(x0=0 if mode == "median" else low, x1=high, fillcolor=ui.ACCENT, opacity=0.07, line_width=0,
              annotation_text="pita lolos", annotation_position="top left")
fig.add_trace(go.Scatter(x=f[sel], y=10 * np.log10(Pr[sel] + 1e-14), name="sebelum", line=dict(color=ui.RAW, width=1.3)))
fig.add_trace(go.Scatter(x=f[sel], y=10 * np.log10(Pc[sel] + 1e-14), name="sesudah", line=dict(color=ui.ACCENT, width=1.8)))
top = 10 * np.log10(max(Pr[sel].max(), Pc[sel].max())) + 5
fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor=ui.PAPER, paper_bgcolor="rgba(0,0,0,0)",
                  legend=dict(orientation="h", y=1.08), xaxis_title="Frekuensi (Hz)", yaxis_title="dB (mV²/Hz)",
                  yaxis_range=[top - 110, top])
fig.update_xaxes(gridcolor="#ecdfe1")
fig.update_yaxes(gridcolor="#ecdfe1")
st.plotly_chart(fig, width="stretch")

st.subheader("Perbandingan konfigurasi pada rekaman ini")
rows = []
for k, lbl in MODES.items():
    nv = float(notch) if k == "bandpass_notch" else None
    y = ui.cached_preprocess(rec.uid, rec, k, low, high, order, nv)
    mm = metrics(rec.uid, rec, y, (k, low, high, order, nv))
    row = {"Konfigurasi": lbl + (f" {notch} Hz" if k == "bandpass_notch" else ""),
           "Baseline (dB)": -mm["baseline_db"], "Jala-jala (dB)": -mm["mains_db"], "> 40 Hz (dB)": -mm["hf_db"]}
    if "snr_after" in mm:
        row["SNR sesudah (dB)"] = mm["snr_after"]
        row["Kemiripan QRS"] = mm["qrs_corr"]
    rows.append(row)
st.dataframe(pd.DataFrame(rows).style.format(precision=2), hide_index=True, width="stretch")
st.caption("Nilai negatif = penurunan daya pada pita derau. SNR dan kemiripan QRS hanya ada bila tersedia sinyal acuan bersih (contoh sintetis). "
           "MIT-BIH direkam di Amerika (60 Hz); rekaman di Indonesia terkena 50 Hz.")
