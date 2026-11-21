"""Titik masuk aplikasi. Jalankan dengan:  streamlit run app.py

st.navigation menentukan nama menu di sidebar (sehingga halaman pertama tampil sebagai "Beranda",
bukan nama berkas "app") dan alamat URL tiap halaman.
"""
import streamlit as st

pages = [
    st.Page("beranda.py", title="Beranda", icon=":material/home:", default=True),
    st.Page("pages/1_Analisis_Sinyal.py", title="Analisis Sinyal", icon=":material/graphic_eq:", url_path="Analisis_Sinyal"),
    st.Page("pages/2_Deteksi_R.py", title="Deteksi R", icon=":material/monitor_heart:", url_path="Deteksi_R"),
    st.Page("pages/3_Klasifikasi.py", title="Klasifikasi", icon=":material/category:", url_path="Klasifikasi"),
    st.Page("pages/5_Tentang.py", title="Tentang", icon=":material/info:", url_path="Tentang"),
]
st.navigation(pages).run()
