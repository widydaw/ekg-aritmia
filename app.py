"""Beranda aplikasi. Jalankan dengan:  streamlit run app.py"""
import numpy as np
import streamlit as st

from src import ui
from src.synthetic import beat_shape, make_profile

ui.page_setup("Beranda")

ui.eyebrow("Pengolahan sinyal digital · MIT-BIH · kelas AAMI")
st.title("Analisis dan Klasifikasi Sinyal EKG untuk Deteksi Aritmia")
st.markdown(
    "Sistem ini menerima sinyal EKG, membersihkannya dengan filter digital, menemukan setiap detak "
    "dengan algoritma Pan-Tompkins, lalu menentukan apakah detak itu **normal (N)**, **prematur "
    "supraventrikular (S)**, atau **prematur ventrikular (V)**. Hasilnya ditampilkan beserta ukuran kinerja metodenya."
)

c1, c2 = st.columns([1.1, 1], gap="large")
with c1:
    st.subheader("Alur pengolahan")
    steps = [
        ("Sinyal EKG", "Deretan angka aktivitas listrik jantung, 360 Hz pada MIT-BIH."),
        ("Praproses", "Band-pass Butterworth 0,5–40 Hz, disaring dua arah."),
        ("Deteksi puncak R", "Pan-Tompkins buatan sendiri. HR = 60 / RR."),
        ("Segmentasi dan fitur", "Segmen −0,2 s s.d. +0,4 s: fitur RR, morfologi, wavelet db4."),
        ("Klasifikasi", "Random Forest, dengan SVM dan k-NN sebagai pembanding."),
        ("Evaluasi", "Data uji dari pasien berbeda (DS1 → DS2)."),
        ("Aplikasi web", "Sinyal, puncak R, HR, kelas tiap detak, dan metrik."),
    ]
    for i, (h, d) in enumerate(steps, 1):
        st.markdown(f"**{i}. {h}** — {d}")
with c2:
    st.subheader("Satu siklus detak")
    p = make_profile(5, hr=75, pA=0.15, pW=0.025, pr=0.17, qA=-0.1, rA=1.2, rW=0.009, sA=-0.25, tA=0.32, tW=0.05, qt=0.4)
    y = np.zeros(int(1.7 * 360))
    beat_shape(y, 0.42, "N", 0.8, p)
    beat_shape(y, 1.22, "N", 0.8, p)
    fig = ui.ecg_figure(0, 1.7, height=280)
    fig.update_xaxes(dtick=0.2, minor=dict(dtick=0.04, showgrid=True))
    ui.add_trace(fig, y, 360, 0, 1.7, width=2)
    tc = 0.42 + (p["qt"] - 0.11) * np.sqrt(0.8)
    for x, yy, t in [(0.42 - p["pr"] + 0.01, 0.32, "P"), (0.39, -0.28, "Q"), (0.42, 1.35, "R"),
                     (0.452, -0.42, "S"), (tc, 0.5, "T"), (1.22, 1.35, "R")]:
        fig.add_annotation(x=x, y=yy, text=f"<b>{t}</b>", showarrow=False, font=dict(color=ui.ACCENT if t == "R" else ui.INK, size=14))
    fig.add_annotation(x=0.82, y=1.62, text="RR = 0,80 s → HR 75 bpm", showarrow=False, font=dict(color=ui.INK2))
    fig.update_yaxes(range=[-0.8, 1.8])
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption("Gelombang P = depolarisasi atrium, QRS = depolarisasi ventrikel, T = repolarisasi ventrikel. "
               "QTc = QT / √RR (Bazett).")

st.subheader("Rumusan masalah")
rm = [
    ("Analisis Sinyal", "Bagaimana merancang filter digital untuk membuang derau EKG, dan seberapa besar perbaikannya?"),
    ("Deteksi R", "Seberapa akurat Pan-Tompkins buatan sendiri dibanding anotasi dokter?"),
    ("Evaluasi Model", "Kelompok fitur apa (RR, morfologi, wavelet) yang paling membedakan N, S, dan V?"),
    ("Evaluasi Model", "Metode mana (Random Forest, SVM, k-NN) yang terbaik pada pembagian per pasien?"),
    ("Klasifikasi", "Bagaimana menyatukan seluruh alur ke aplikasi web yang akurat dan cukup cepat?"),
]
cols = st.columns(5)
for i, (col, (page, q)) in enumerate(zip(cols, rm), 1):
    with col.container(border=True):
        st.markdown(f"**RM {i}**")
        st.write(q)
        st.caption(f"Lihat: {page}")

st.subheader("Cara memakai")
st.markdown(
    "1. Pilih sumber rekaman di panel kiri: rekaman **MIT-BIH** (diunduh dari PhysioNet), **contoh sintetis**, "
    "atau **unggah CSV** satu kolom beserta frekuensi samplingnya.\n"
    "2. Buka **Analisis Sinyal** untuk membandingkan sinyal asli, sinyal bersih, dan spektrumnya.\n"
    "3. Buka **Deteksi R** untuk melihat puncak R, HR, dan tahapan Pan-Tompkins.\n"
    "4. Buka **Klasifikasi** untuk melihat kelas tiap detak dan mengunduh hasilnya, lalu **Evaluasi Model** "
    "untuk kinerja pada pasien yang belum pernah dilihat model."
)

if not ui.load_models():
    ui.models_missing_message()
st.caption("Untuk pembelajaran, bukan alat diagnosis medis.")
