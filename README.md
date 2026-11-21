# Analisis Sinyal EKG untuk Deteksi Aritmia

Aplikasi Streamlit yang menerima sinyal EKG, menyaringnya dengan filter digital, lalu mendeteksi
puncak R dengan Pan-Tompkins buatan sendiri untuk menghitung detak jantung (HR) dan variabilitasnya.

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
streamlit run app.py         # buka http://localhost:8501
pip install -r requirements-dev.txt && pytest   # uji fungsi tanpa membuka web
```

Untuk membagikan aplikasi lewat internet (Streamlit Community Cloud), ikuti [DEPLOY.md](DEPLOY.md).

## Halaman

| Halaman | Isi |
|---|---|
| Beranda | Ringkasan, alur pengolahan, rumusan masalah, petunjuk |
| Analisis Sinyal | Sinyal asli dan bersih, spektrum Welch, perbandingan konfigurasi filter, keterangan notch 50/60 Hz (RM 1) |
| Deteksi R | Puncak R, HR, SDNN, RMSSD, Se/PPV terhadap anotasi, monitor berjalan, tahapan Pan-Tompkins (RM 2) |
| Tentang | Dataset, format berkas, metode, data latihan, rujukan |

## Sumber rekaman

Dipilih di panel kiri:

- **MIT-BIH** — 44 rekaman, diunduh dari PhysioNet saat dipilih.
- **Contoh sintetis** — 6 rekaman 60 detik dengan anotasi dan sinyal bersih acuan.
- **Unggah berkas** — format standar EKG:

| Format | Berkas | Catatan |
|---|---|---|
| WFDB (PhysioNet) | `.hea` + `.dat` atau `.mat`, opsional `.atr` / `.qrs` | Unggah sekaligus; anotasi dipakai untuk Se/PPV |
| EDF / EDF+ | `.edf` | fs dan satuan dari header |
| HL7 aECG | `.xml` | Standar FDA |
| MATLAB | `.mat` | Isi fs manual |
| CSV / TXT | `.csv`, `.txt` | Satu kolom per sadapan; kolom waktu dikenali, fs bisa dihitung darinya |

Sinyal diubah ke mV, sadapan bisa dipilih (II/MLII diutamakan), lalu disampel ulang ke 360 Hz.
Rekaman 10 detik sampai 30 menit; yang lebih panjang dipotong ke 30 menit pertama.

## Struktur

```
app.py                  navigasi halaman
beranda.py              Beranda
pages/                  Analisis Sinyal, Deteksi R, Tentang
src/
  data_loader.py        MIT-BIH (wfdb), anotasi, sampel ulang
  file_formats.py       pembaca WFDB, EDF/EDF+, HL7 aECG, MAT, CSV/TXT
  preprocessing.py      band-pass, notch, median; spektrum dan metrik filter
  qrs_detection.py      Pan-Tompkins, pencocokan dengan anotasi, HR/HRV
  monitor.py            monitor EKG berjalan
  synthetic.py          sinyal sintetis untuk demo dan tes
  ui.py                 komponen bersama Streamlit (cache, pemilih rekaman, grafik)
tests/                  pengujian fungsi dan format berkas
```

Berkas klasifikasi lama (`train.py`, `src/features.py`, `src/classifier.py`, `src/evaluation.py`,
`models/`) tidak lagi dipakai oleh aplikasi web.

Antarmuka fungsi:

```python
preprocess(sinyal, fs)                 # -> sinyal bersih
detect_r_peaks(sinyal_bersih, fs)      # -> indeks puncak R
read_upload([(nama, bytes), ...], fs)  # -> semua sadapan berkas unggahan
```
