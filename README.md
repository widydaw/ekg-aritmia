# Analisis dan Klasifikasi Sinyal EKG untuk Deteksi Aritmia

Aplikasi Streamlit yang menerima sinyal EKG, menyaringnya dengan filter digital, mendeteksi puncak R
dengan Pan-Tompkins buatan sendiri, mengekstraksi fitur RR, morfologi, dan wavelet, lalu
mengklasifikasikan tiap detak ke kelas AAMI **N**, **S**, atau **V**.

> Untuk pembelajaran, bukan alat diagnosis medis.

## Pemasangan

Butuh Python 3.10 atau lebih baru.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Di Windows dengan **Smart App Control** aktif, versi pustaka yang sangat baru bisa diblokir
("An Application Control policy has blocked this file"). Versi di `requirements.txt` sudah diuji
lolos. Jika path folder proyek sangat panjang, buat `.venv` di path pendek (misalnya `C:\venv\ekg`).

## Menjalankan

```bash
python train.py              # unduh MIT-BIH (±100 MB ke data/mitdb), latih DS1, uji DS2
streamlit run app.py         # buka http://localhost:8501
```

Pilihan lain:

```bash
python train.py --tune       # + setel parameter Random Forest dengan GroupKFold per rekaman
python train.py --synthetic  # data sintetis, beberapa detik, tanpa internet (hanya uji coba)
pip install -r requirements-dev.txt && pytest   # uji fungsi tanpa membuka web
```

Untuk membagikan aplikasi lewat internet (Streamlit Community Cloud), ikuti [DEPLOY.md](DEPLOY.md).

`train.py` menyimpan `models/rf.joblib`, `svm.joblib`, `knn.joblib`, dan `models/results.json`
(confusion matrix, metrik per kelas, perbandingan model dan kelompok fitur, pembagian acak vs per
pasien, serta evaluasi end-to-end dengan detektor sendiri). Aplikasi web hanya memuat berkas ini.

## Halaman

| Halaman | Isi |
|---|---|
| Beranda | Ringkasan, alur pengolahan, rumusan masalah, petunjuk |
| Analisis Sinyal | Sinyal asli dan bersih, spektrum Welch, perbandingan konfigurasi filter (RM 1) |
| Deteksi R | Puncak R, HR, SDNN, RMSSD, Se/PPV terhadap anotasi, tahapan Pan-Tompkins (RM 2) |
| Klasifikasi | Detak berwarna N/S/V, bentuk detak per kelas, tabel, unduh CSV (RM 5) |
| Evaluasi Model | Confusion matrix, metrik per kelas, RF vs SVM vs k-NN, kelompok fitur, fitur penting (RM 3, RM 4) |
| Tentang | Dataset, metode, keterbatasan, rujukan |

Sumber rekaman dipilih di panel kiri: **MIT-BIH** (diunduh per rekaman bila belum ada),
**contoh sintetis** (label acuan dan sinyal bersih diketahui), atau **unggah CSV** satu kolom
dengan frekuensi sampling 50–2000 Hz, 10 detik sampai 30 menit (disampel ulang ke 360 Hz).

## Struktur

```
app.py                  Beranda
pages/                  Analisis, Deteksi R, Klasifikasi, Evaluasi, Tentang
train.py                pelatihan dan evaluasi DS1 → DS2
src/
  data_loader.py        MIT-BIH (wfdb), kelas AAMI, DS1/DS2, CSV
  preprocessing.py      band-pass, notch, median; spektrum dan metrik filter
  qrs_detection.py      Pan-Tompkins, pencocokan dengan anotasi, HR/HRV
  features.py           segmentasi dan 38 fitur (RR, morfologi, wavelet db4, relatif pasien)
  classifier.py         RF, SVM, k-NN; simpan dan muat dengan joblib
  evaluation.py         confusion matrix dan metrik per kelas
  synthetic.py          sinyal sintetis untuk demo, tes, dan fungsi tiruan
  ui.py                 komponen bersama Streamlit (cache, pemilih rekaman, grafik)
models/                 model terlatih dan results.json
data/                   rekaman MIT-BIH (tidak di-commit)
tests/                  pengujian fungsi
```

Antarmuka fungsi yang disepakati:

```python
preprocess(sinyal, fs)                          # -> sinyal bersih
detect_r_peaks(sinyal_bersih, fs)               # -> indeks puncak R
extract_features(sinyal_bersih, puncak_r, fs)   # -> matriks fitur (n_detak × 30)
predict(model, fitur)                           # -> label kelas
evaluate(label_benar, label_prediksi)           # -> confusion matrix dan metrik per kelas
```

## Catatan metode

- Kanal MLII; rekaman alat pacu 102, 104, 107, 217 dibuang. Kelas F dan Q tidak diklasifikasi.
- Fitur latih diambil pada posisi R dari anotasi; di web dan evaluasi end-to-end dipakai detektor sendiri.
- Fitur relatif pasien (8): RR dibagi rata-rata rekaman, korelasi detak/QRS/daerah P dengan detak median
  pasien, selisih daerah P, rasio amplitudo R, dan selisih lebar QRS. Fitur ini membantu kelas S pada
  pembagian per pasien. `extract_features` harus dipanggil per rekaman utuh.
- Bobot keputusan per kelas (kelas = argmax probabilitas × bobot) disetel dengan GroupKFold per
  rekaman pada DS1 saja, lalu dipakai apa adanya di DS2. Matikan dengan `--no-weights`.
- Hasil MIT-BIH saat ini (DS2): Random Forest F1 makro 0,787; kelas S sensitivitas 88,8%,
  presisi 30,1%. Tanpa fitur relatif dan bobot: F1 makro 0,681, sensitivitas S 7,6%.
- Bobot kelas seimbang. SVM dilatih pada semua detak S dan V serta maksimal 8.000 detak N
  (`--svm-cap`) karena biayanya kuadratik.
- Cocokkan kembali daftar DS1/DS2 dan sitasi dengan sumber aslinya sebelum dipakai di laporan.
