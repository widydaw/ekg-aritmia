import streamlit as st

from src import ui
from src.file_formats import FORMATS

ui.page_setup("Tentang")
ui.eyebrow("Tentang")
st.title("Dataset, format berkas, dan metode")
st.warning("**Bukan alat diagnosis medis.** Aplikasi ini dibuat untuk pembelajaran pengolahan sinyal digital. "
           "Hasilnya tidak boleh dipakai untuk keputusan klinis.")

c1, c2 = st.columns(2, gap="large")
with c1:
    st.subheader("Dataset bawaan")
    st.write("MIT-BIH Arrhythmia Database (PhysioNet): rekaman sekitar 30 menit, 360 Hz, dua kanal, dengan anotasi "
             "posisi setiap detak dari dokter. Aplikasi memakai kanal MLII dan menyediakan 44 rekaman "
             "(rekaman alat pacu 102, 104, 107, dan 217 tidak disertakan).")
    st.subheader("Metode")
    st.write("Band-pass Butterworth orde 4, 0,5–40 Hz dengan penyaringan dua arah (sosfiltfilt), pilihan notch 50/60 Hz "
             "dan filter median; deteksi puncak R dengan Pan-Tompkins (band-pass 5–15 Hz, turunan, kuadrat, integrasi "
             "150 ms, ambang adaptif, refrakter 200 ms, cek gelombang T, pencarian mundur); HR = 60 / RR, SDNN, dan RMSSD.")
    st.write("Klasifikasi: 38 fitur per detak (6 fitur RR, 14 morfologi, 10 wavelet db4 level 4, 8 fitur relatif "
             "terhadap detak median pasien). Model Random Forest, SVM RBF, dan k-NN dilatih pada DS1 dan diuji pada DS2 "
             "(pembagian per pasien menurut de Chazal dkk., 2004), dengan bobot keputusan per kelas yang disetel "
             "hanya pada DS1. Kelas mengikuti standar AAMI: N, S, V (F dan Q tidak diklasifikasi).")
    st.write("Ringkasan irama (berbasis aturan): detak prematur sering bila S+V ≥ 1% detak; bradikardia bila HR < 60 bpm, "
             "takikardia bila > 100 bpm; keteraturan dinilai per 30 interval RR (RR di sekitar detak V dibuang), "
             "potongan tidak teratur bila median selisih RR berturutan > 8% median RR, dan curiga fibrilasi atrium bila "
             "≥ 50% potongan tidak teratur.")
    st.subheader("Keterbatasan")
    st.write("Rekaman yang sangat berderau atau berisi banyak artefak gerakan dapat menurunkan akurasi deteksi. "
             "Rekaman lebih dari 30 menit hanya dianalisis 30 menit pertamanya. Contoh sintetis hanya untuk demo. "
             "Model hanya dilatih pada sadapan MLII MIT-BIH; pada sadapan atau alat rekam lain hasil klasifikasi bisa "
             "kurang tepat, dan detak S masih sering tertukar dengan N.")
with c2:
    st.subheader("Format berkas yang bisa diunggah")
    st.markdown("\n".join(f"- **{k}** — {v}" for k, v in FORMATS.items()))
    st.caption("Semua sinyal diubah ke mV lalu disampel ulang ke 360 Hz. Rekaman 10 detik sampai 30 menit; "
               "ukuran berkas maksimal 50 MB. Bila berkas WFDB disertai .atr, aplikasi menghitung sensitivitas "
               "dan presisi deteksi terhadap anotasi.")
    st.write("Format standar lain seperti SCP-ECG dan DICOM-ECG belum didukung; ubah dulu ke EDF atau WFDB.")

st.subheader("Data aritmia untuk latihan")
st.markdown("""
| Dataset | Isi | Format | Bisa langsung diunggah |
| --- | --- | --- | --- |
| [MIT-BIH Arrhythmia](https://physionet.org/content/mitdb/1.0.0/) | 48 rekaman 30 menit, 360 Hz, anotasi tiap detak | WFDB (.hea, .dat, .atr) | Ya, unggah ketiga berkas sekaligus |
| [MIT-BIH Atrial Fibrillation](https://physionet.org/content/afdb/1.0.0/) | 25 rekaman ±10 jam, 250 Hz, episode fibrilasi atrium | WFDB (.hea, .dat, .qrs) | Ya, 30 menit pertama dianalisis |
| [Long Term AF Database](https://physionet.org/content/ltafdb/1.0.0/) | 84 rekaman 24–25 jam, 128 Hz, AF paroksismal | WFDB (.hea, .dat, .atr) | Ya, 30 menit pertama dianalisis |
| [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) | 21.837 EKG 12 sadapan, 10 detik, 100/500 Hz | WFDB (.hea, .dat) | Ya, pilih sadapan II |
| [Chapman–Shaoxing](https://doi.org/10.6084/m9.figshare.c.4560497) | 10.646 EKG 12 sadapan, 10 detik, 500 Hz, 11 jenis irama | CSV (12 kolom) | Ya, isi fs 500 Hz |
| [PhysioNet Challenge 2017](https://physionet.org/content/challenge-2017/1.0.0/) | 8.528 EKG satu sadapan, 9–60 detik, 300 Hz, AF vs normal | WFDB (.hea, .mat) | Ya, unggah .hea dan .mat |
| [PhysioNet Challenge 2020](https://physionet.org/content/challenge-2020/1.0.2/) | 43.101 EKG 12 sadapan dari 4 negara, 27 diagnosis | WFDB (.hea, .mat) | Ya, unggah .hea dan .mat |
""")

st.subheader("Rujukan")
st.markdown("""
1. Pan, J. dan Tompkins, W.J. (1985). A Real-Time QRS Detection Algorithm. *IEEE Trans. Biomed. Eng.*, 32(3), 230–236.
2. Moody, G.B. dan Mark, R.G. (2001). The impact of the MIT-BIH arrhythmia database. *IEEE Eng. Med. Biol. Mag.*, 20(3), 45–50. [doi:10.1109/51.932724](https://doi.org/10.1109/51.932724)
3. Goldberger, A.L. dkk. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23), e215–e220. [doi:10.1161/01.cir.101.23.e215](https://doi.org/10.1161/01.cir.101.23.e215)
4. Petrutiu, S., Sahakian, A.V., dan Swiryn, S. (2007). Abrupt changes in fibrillatory wave characteristics at the termination of paroxysmal atrial fibrillation in humans. *Europace*, 9(7), 466–470. [doi:10.1093/europace/eum096](https://doi.org/10.1093/europace/eum096)
5. Wagner, P. dkk. (2020). PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*, 7, 154. [doi:10.1038/s41597-020-0495-6](https://doi.org/10.1038/s41597-020-0495-6)
6. Zheng, J. dkk. (2020). A 12-lead electrocardiogram database for arrhythmia research covering more than 10,000 patients. *Scientific Data*, 7, 48. [doi:10.1038/s41597-020-0386-x](https://doi.org/10.1038/s41597-020-0386-x)
7. Clifford, G.D. dkk. (2017). AF Classification from a Short Single Lead ECG Recording: the PhysioNet/Computing in Cardiology Challenge 2017. *Computing in Cardiology*, 44.
8. Perez Alday, E.A. dkk. (2020). Classification of 12-lead ECGs: the PhysioNet/Computing in Cardiology Challenge 2020. *Physiological Measurement*, 41(12), 124003. [doi:10.1088/1361-6579/abc960](https://doi.org/10.1088/1361-6579/abc960)
9. de Chazal, P., O'Dwyer, M., dan Reilly, R.B. (2004). Automatic classification of heartbeats using ECG morphology and heartbeat interval features. *IEEE Trans. Biomed. Eng.*, 51(7), 1196–1206. [doi:10.1109/TBME.2004.827359](https://doi.org/10.1109/TBME.2004.827359)
10. ANSI/AAMI EC57:2012. Testing and reporting performance results of cardiac rhythm and ST segment measurement algorithms.
""")
st.caption("Cek kembali rincian sitasi pada sumber aslinya sebelum dipakai di laporan.")
