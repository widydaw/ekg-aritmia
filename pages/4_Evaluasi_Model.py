import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.classifier import MODEL_NAMES
from src.data_loader import CLASS_NAMES, CLASSES

ui.page_setup("Evaluasi Model")
ui.eyebrow("RM 3 & 4 · Evaluasi")
st.title("Kinerja model pada pasien yang belum pernah dilihat")
st.write("Model dilatih pada DS1 dan diuji pada DS2 (pasien berbeda), mengikuti skema inter-patient de Chazal dkk. (2004). "
         "Fitur latih diambil pada posisi R dari anotasi agar kinerja klasifikasi terukur terpisah dari kesalahan deteksi.")

R = ui.load_results()
if R is None:
    ui.models_missing_message()
    st.stop()
meta = R["meta"]
if meta["source"] != "MIT-BIH":
    st.warning("Hasil ini berasal dari data **sintetis** (`train.py --synthetic`). Jangan dikutip di laporan; "
               "jalankan `python train.py` untuk hasil MIT-BIH.")
st.caption(f"Sumber: {meta['source']} · dibuat {meta['created']} · scikit-learn {meta['sklearn']} · "
           f"DS1 {R['counts']['DS1']} · DS2 {R['counts']['DS2']}")

key = st.radio("Model", list(R["models"]), format_func=MODEL_NAMES.get, horizontal=True)
ev = R["models"][key]
per = pd.DataFrame(ev["per_class"])
f1s = per.set_index("kelas")
c = st.columns(4)
c[0].metric("F1 makro", f"{ev['macro_f1']:.3f}", help="Acuan utama")
c[1].metric("Akurasi", f"{ev['accuracy']:.1%}", help="Pelengkap saja; didominasi kelas N")
c[2].metric("Sensitivitas S", f"{f1s.loc['S', 'sensitivitas']:.1%}")
c[3].metric("Sensitivitas V", f"{f1s.loc['V', 'sensitivitas']:.1%}", f"PPV {f1s.loc['V', 'presisi']:.1%}", delta_color="off")

l, r = st.columns(2, gap="large")
with l:
    st.subheader("Confusion matrix")
    cm = pd.DataFrame(ev["cm"], index=[f"acuan {k}" for k in CLASSES], columns=[f"pred {k}" for k in CLASSES])
    share = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
    fig = go.Figure(go.Heatmap(z=share.values, x=cm.columns, y=cm.index, colorscale=[[0, "#fffdfc"], [1, ui.ACCENT]],
                               zmin=0, zmax=1, showscale=False,
                               text=[[f"{v:,}<br>{s:.0%}" for v, s in zip(rv, rs)] for rv, rs in zip(cm.values, share.values)],
                               texttemplate="%{text}", hoverinfo="skip"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption("Baris = label acuan, kolom = prediksi. Persentase per baris.")
with r:
    st.subheader("Metrik per kelas")
    tbl = per[["kelas", "jumlah", "sensitivitas", "presisi", "spesifisitas", "f1"]].copy()
    tbl["kelas"] = tbl["kelas"].map(lambda k: f"{k} · {CLASS_NAMES[k]}")
    st.dataframe(tbl, hide_index=True, width="stretch",
                 column_config={k: st.column_config.NumberColumn(format="%.3f") for k in ["sensitivitas", "presisi", "spesifisitas", "f1"]})
    st.caption("Sensitivitas = TP/(TP+FN), presisi = TP/(TP+FP), spesifisitas = TN/(TN+FP), F1 = 2·P·R/(P+R).")

l, r = st.columns(2, gap="large")
with l:
    st.subheader("RM 4 · Perbandingan metode")
    rows = []
    for k, e in R["models"].items():
        d = {"Metode": MODEL_NAMES[k]}
        d.update({f"F1 {p['kelas']}": p["f1"] for p in e["per_class"]})
        d.update({"F1 makro": e["macro_f1"], "Akurasi": e["accuracy"], "Waktu latih (s)": e["train_seconds"], "Detak latih": e["n_train"]})
        rows.append(d)
    dfm = pd.DataFrame(rows)
    fig = go.Figure()
    for k in CLASSES:
        fig.add_trace(go.Bar(name=f"F1 {k}", x=dfm["Metode"], y=dfm[f"F1 {k}"], marker_color=ui.CLASS_COLORS[k]))
    fig.add_trace(go.Scatter(name="F1 makro", x=dfm["Metode"], y=dfm["F1 makro"], mode="markers+text", text=[f"{v:.3f}" for v in dfm["F1 makro"]],
                             textposition="top center", marker=dict(color=ui.INK, size=11, symbol="diamond")))
    fig.update_layout(barmode="group", height=320, margin=dict(l=10, r=10, t=30, b=10), yaxis_range=[0, 1.1],
                      legend=dict(orientation="h", y=1.15), plot_bgcolor=ui.PAPER)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(dfm, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in dfm.columns if c.startswith("F1") or c == "Akurasi"})
    st.caption("SVM dilatih pada sebagian detak N (semua S dan V) karena biaya komputasinya kuadratik.")
    bw = {MODEL_NAMES[k]: e.get("bobot") for k, e in R["models"].items() if e.get("bobot")}
    if bw:
        st.markdown("**Bobot keputusan per kelas** (kelas = argmax probabilitas × bobot), disetel dengan "
                    "GroupKFold per rekaman hanya pada DS1:")
        st.dataframe(pd.DataFrame([{"Metode": k, "Bobot N": v["weights"][0], "Bobot S": v["weights"][1], "Bobot V": v["weights"][2],
                                    "F1 makro CV DS1 tanpa bobot": v["cv_f1_macro_tanpa_bobot"], "F1 makro CV DS1 dg. bobot": v["cv_f1_macro"],
                                    "F1 makro DS2 tanpa bobot": R["models"][[m for m in MODEL_NAMES if MODEL_NAMES[m] == k][0]]["macro_f1_tanpa_bobot"]}
                                   for k, v in bw.items()]),
                     hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in
                                    ["F1 makro CV DS1 tanpa bobot", "F1 makro CV DS1 dg. bobot", "F1 makro DS2 tanpa bobot"]})
    if R.get("ablation_rf"):
        st.markdown("**Ablasi Random Forest** (F1 makro DS2)")
        st.dataframe(pd.Series(R["ablation_rf"], name="F1 makro").rename_axis("Konfigurasi").reset_index(),
                     hide_index=True, width="stretch", column_config={"F1 makro": st.column_config.NumberColumn(format="%.3f")})
with r:
    st.subheader("RM 3 · Kelompok fitur")
    g = pd.Series(R["feature_groups"])
    fig = go.Figure(go.Bar(x=g.values, y=g.index, orientation="h", marker_color=ui.ACCENT, text=[f"{v:.3f}" for v in g.values], textposition="outside"))
    fig.update_layout(height=250, margin=dict(l=10, r=40, t=10, b=10), xaxis=dict(range=[0, 1.1], title="F1 makro DS2 (Random Forest)"),
                      yaxis=dict(autorange="reversed"), plot_bgcolor=ui.PAPER)
    st.plotly_chart(fig, width="stretch")
    st.subheader("Fitur paling berpengaruh")
    imp = pd.DataFrame(R["importance"]).sort_values("nilai", ascending=False).head(12)
    colors = {"RR": ui.CLASS_COLORS["S"], "Morfologi": ui.CLASS_COLORS["N"], "Wavelet": ui.CLASS_COLORS["V"], "Relatif pasien": ui.ACCENT}
    fig = go.Figure(go.Bar(x=imp["nilai"], y=imp["fitur"], orientation="h", marker_color=imp["kelompok"].map(colors),
                           hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"),
                      xaxis=dict(tickformat=".0%", title="Penurunan Gini (Random Forest)"), plot_bgcolor=ui.PAPER)
    st.plotly_chart(fig, width="stretch")
    st.caption("Warna: kuning = RR, hijau toska = morfologi, ungu = wavelet, merah = relatif pasien.")

l, r = st.columns(2, gap="large")
with l:
    st.subheader("Mengapa pembagian per pasien?")
    s = R["split"]
    st.dataframe(pd.DataFrame({"Pembagian": ["Acak per detak", "Per pasien (DS1 → DS2)"],
                               "F1 makro": [s["acak_macro_f1"], s["pasien_macro_f1"]],
                               "F1 kelas S": [s["acak_f1_S"], s["pasien_f1_S"]]}),
                 hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["F1 makro", "F1 kelas S"]})
    st.write("Pembagian acak memasukkan detak pasien yang sama ke data latih dan uji, sehingga hasilnya tampak terlalu bagus. "
             "Pembagian per pasien lebih jujur karena pasien uji belum pernah dilihat model.")
with r:
    st.subheader("Evaluasi menyeluruh (end-to-end)")
    e = R["end_to_end"]
    c = st.columns(3)
    c[0].metric("Se deteksi R", f"{e['sensitivitas']:.2%}")
    c[1].metric("PPV deteksi R", f"{e['presisi']:.2%}")
    c[2].metric("F1 makro", f"{e['evaluasi']['macro_f1']:.3f}", f"{e['evaluasi']['macro_f1'] - R['models']['rf']['macro_f1']:+.3f} vs anotasi")
    st.caption("Posisi R berasal dari detektor Pan-Tompkins sendiri. Selisihnya menunjukkan pengaruh kesalahan deteksi.")
    with st.expander("Kinerja deteksi per rekaman DS2"):
        st.dataframe(pd.DataFrame(e["per_rekaman"]), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.4f") for c in ["sensitivitas", "presisi"]})

if R.get("tuning"):
    st.caption(f"Parameter RF hasil penyetelan GroupKFold: {R['tuning']['best_params']} (F1 makro CV {R['tuning']['best_cv_f1_macro']:.3f}).")
