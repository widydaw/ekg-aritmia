import streamlit as st

from src import ui

ui.page_setup("Tentang")
ui.eyebrow("Tentang")
st.title("Dataset, metode, dan keterbatasan")
st.warning("**Bukan alat diagnosis medis.** Aplikasi ini dibuat untuk pembelajaran pengolahan sinyal digital. "
           "Hasilnya tidak boleh dipakai untuk keputusan klinis.")

c1, c2 = st.columns(2, gap="large")
with c1:
    st.subheader("Dataset")
    st.write("MIT-BIH Arrhythmia Database (PhysioNet): 48 rekaman sekitar 30 menit dari 47 subjek, 360 Hz, dua kanal, "
             "sekitar 110 ribu detak yang dianotasi dokter. Proyek memakai kanal MLII dan membuang rekaman alat pacu "
             "102, 104, 107, dan 217. Kelas AAMI: N (N, L, R, e, j), S (A, a, J, S), V (V, E); F dan Q tidak diklasifikasi.")
    st.subheader("Metode")
    st.write("Band-pass Butterworth orde 4, 0,5–40 Hz dengan sosfiltfilt; Pan-Tompkins dengan ambang adaptif, refrakter 200 ms, "
             "cek gelombang T, dan pencarian mundur; 38 fitur per detak (6 RR, 14 morfologi, 10 wavelet db4 empat level, "
             "8 relatif terhadap pasien: RR dibagi rata-rata rekaman dan kemiripan dengan detak median pasien); "
             "Random Forest dengan bobot kelas seimbang, serta SVM RBF dan k-NN dengan fitur ternormalisasi. "
             "Bobot keputusan per kelas disetel dengan GroupKFold per rekaman pada DS1 saja.")
with c2:
    st.subheader("Keterbatasan")
    st.write("Kelas S biasanya paling sulit pada pembagian per pasien karena bentuknya mirip normal dan jumlahnya sedikit. "
             "Rekaman yang sangat berderau dapat menurunkan akurasi deteksi. Contoh sintetis hanya untuk demo dan pengujian.")
    st.subheader("Format CSV unggahan")
    st.write("Satu kolom nilai sinyal (mV disarankan), boleh dengan satu baris judul. Isi frekuensi sampling. "
             "Minimal 10 detik, maksimal 30 menit. Sinyal disampel ulang ke 360 Hz agar cocok dengan model.")

st.subheader("Rujukan awal")
st.markdown("""
1. Pan, J. dan Tompkins, W.J. (1985). A Real-Time QRS Detection Algorithm. *IEEE Transactions on Biomedical Engineering*, 32(3), 230–236.
2. Moody, G.B. dan Mark, R.G. (2001). The impact of the MIT-BIH Arrhythmia Database. *IEEE Engineering in Medicine and Biology Magazine*, 20(3), 45–50.
3. de Chazal, P., O'Dwyer, M., dan Reilly, R.B. (2004). Automatic classification of heartbeats using ECG morphology and heartbeat interval features. *IEEE Transactions on Biomedical Engineering*, 51(7), 1196–1206.
4. Luz, E.J.S., Schwartz, W.R., Cámara-Chávez, G., dan Menotti, D. (2016). ECG-based heartbeat classification for arrhythmia detection: A survey. *Computer Methods and Programs in Biomedicine*, 127, 144–164.
5. Saito, T. dan Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. *PLoS ONE*, 10(3), e0118432.
6. ANSI/AAMI EC57:2012. Testing and reporting performance results of cardiac rhythm and ST segment measurement algorithms.
""")
st.caption("Cek kembali rincian sitasi dan daftar rekaman DS1 dan DS2 pada sumber aslinya sebelum dipakai di laporan.")
