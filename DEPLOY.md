# Membagikan aplikasi lewat internet (Streamlit Community Cloud)

Hasil akhirnya: link seperti `https://ekg-aritmia.streamlit.app` yang bisa dibuka dari HP atau
laptop mana saja, tanpa laptop Anda harus menyala. Gratis. Tidak perlu memasang Git.

## 1. Siapkan berkas (sekali, dan setiap kali kode atau model berubah)

Di folder **Documents\Proyek EKG Aritmia**, klik dua kali **`Siapkan Unggahan GitHub.bat`**.
Folder **`Unggah ke GitHub`** akan dibuat dan terbuka. Isinya ±27 berkas (±26 MB): kode,
model terlatih, dan pengaturan, tanpa data MIT-BIH (server mengunduh rekaman sendiri dari PhysioNet).

## 2. Buat repositori di GitHub

1. Buka https://github.com dan masuk (buat akun dulu bila belum punya).
2. Klik **New repository** (tombol **+** di kanan atas → *New repository*).
3. Isi **Repository name**, misalnya `ekg-aritmia`. Pilih **Public**
   (Streamlit Community Cloud gratis membaca repositori publik).
4. Jangan centang "Add a README". Klik **Create repository**.

## 3. Unggah berkas

1. Di halaman repositori yang baru, klik tautan **uploading an existing file**.
2. Buka folder **`Unggah ke GitHub`**, pilih **semua isinya** (Ctrl+A), lalu seret ke halaman GitHub.
   Seret *isi* foldernya, bukan folder itu sendiri, supaya `app.py` ada di tingkat paling atas.
3. Tunggu sampai semua berkas selesai terunggah, lalu klik **Commit changes**.
4. Periksa: di halaman repositori harus terlihat `app.py`, `requirements.txt`, folder `pages`,
   `src`, `models`, dan `.streamlit`.
   - Bila folder **`.streamlit`** tidak ikut (beberapa browser melewatkan folder berawalan titik):
     klik **Add file → Create new file**, ketik nama `.streamlit/config.toml`, salin isi berkas
     `.streamlit\config.toml` dari folder unggahan, lalu **Commit changes**. Tanpa berkas ini aplikasi
     tetap jalan, hanya warnanya memakai tema bawaan Streamlit.

## 4. Pasang di Streamlit Community Cloud

1. Buka https://share.streamlit.io dan masuk dengan **Continue with GitHub**.
2. Klik **Create app** → **Deploy a public app from GitHub**.
3. Isi:
   - **Repository**: `nama-akun-anda/ekg-aritmia`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: pilih nama, misalnya `ekg-aritmia`
4. Buka **Advanced settings** → **Python version**: pilih **3.12**. Klik **Save**.
5. Klik **Deploy**. Pemasangan pertama butuh beberapa menit.

Setelah selesai, bagikan link `https://<nama>.streamlit.app` ke dosen atau anggota kelompok.

## Memperbarui aplikasi

Setelah kode atau model berubah: jalankan lagi `Siapkan Unggahan GitHub.bat`, lalu di GitHub klik
**Add file → Upload files**, seret isi folder yang sama (berkas lama tertimpa), dan **Commit changes**.
Streamlit Cloud memasang ulang aplikasi secara otomatis.

## Bila ada masalah

| Gejala | Penyebab dan cara mengatasi |
|---|---|
| Gagal saat "Installing requirements" | Ganti Python version ke 3.13 di *Settings → General*, lalu *Reboot app*. |
| "Model belum ada" | Folder `models` (berisi `.joblib`) belum terunggah. Unggah ulang. |
| Rekaman MIT-BIH gagal dimuat | PhysioNet sedang sibuk. Coba lagi beberapa saat, atau pakai contoh sintetis. |
| Aplikasi "tertidur" | Aplikasi gratis tidur bila lama tidak dibuka. Klik "Yes, get this app back up". |

Catatan: versi scikit-learn di `requirements.txt` (1.8.0) harus sama dengan versi saat model
dilatih. Bila Anda melatih ulang dengan versi lain, perbarui `requirements.txt` juga.
Aplikasi ini untuk pembelajaran, bukan alat diagnosis medis.
