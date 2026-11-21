"""Membaca rekaman MIT-BIH dan anotasinya, pemetaan kelas AAMI, pembagian DS1/DS2,
serta membaca rekaman sendiri dari berkas CSV satu kolom."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import hashlib

import numpy as np
from scipy.signal import resample_poly

FS_MITBIH = 360
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "mitdb"

# Rekaman dengan alat pacu jantung dibuang (kebiasaan standar AAMI).
PACED = [102, 104, 107, 217]

# Pembagian per pasien menurut de Chazal dkk. (2004).
# Cocokkan kembali dengan makalah acuan sebelum dipakai di laporan.
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
       201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210,
       212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

# Pemetaan simbol anotasi MIT-BIH ke kelas AAMI.
AAMI = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}
CLASSES = ["N", "S", "V"]
CLASS_NAMES = {"N": "Normal", "S": "Supraventrikular prematur", "V": "Ventrikular prematur"}


@dataclass
class Record:
    """Satu rekaman EKG beserta (bila ada) anotasi detaknya."""
    uid: str                      # kunci unik untuk cache
    name: str                     # nama yang ditampilkan
    signal: np.ndarray            # sinyal dalam mV
    fs: float                     # frekuensi sampling (Hz)
    r_peaks: np.ndarray | None = None   # indeks sampel detak beranotasi
    labels: np.ndarray | None = None    # kelas AAMI tiap detak (N, S, V, F, Q)
    lead: str = "MLII"
    clean_reference: np.ndarray | None = None  # hanya untuk sinyal sintetis
    description: str = ""

    @property
    def duration(self) -> float:
        return len(self.signal) / self.fs

    @property
    def has_annotations(self) -> bool:
        return self.r_peaks is not None and self.labels is not None


MITDB_DAT_BYTES = 1_950_000  # 650.000 sampel × 2 kanal × 12 bit


def _complete(name: str, data_dir: Path) -> bool:
    dat = data_dir / f"{name}.dat"
    return (all((data_dir / f"{name}.{ext}").exists() for ext in ("hea", "atr"))
            and dat.exists() and dat.stat().st_size == MITDB_DAT_BYTES)


def download_mitdb(records=None, data_dir: Path = DATA_DIR, retries: int = 5) -> Path:
    """Unduh rekaman MIT-BIH dari PhysioNet ke data/mitdb (hanya yang belum lengkap).

    Tiap rekaman dicoba ulang beberapa kali bila koneksi terputus.
    """
    import time

    import wfdb

    data_dir.mkdir(parents=True, exist_ok=True)
    names = [str(r) for r in (records if records is not None else DS1 + DS2)]
    for n in names:
        for attempt in range(1, retries + 1):
            if _complete(n, data_dir):
                break
            try:
                wfdb.dl_database("mitdb", dl_dir=str(data_dir), records=[n])
            except Exception as e:  # koneksi putus, server sibuk
                if attempt == retries:
                    raise RuntimeError(f"Rekaman {n} gagal diunduh setelah {retries} percobaan: {e}") from e
                print(f"  rekaman {n}: unduhan gagal ({type(e).__name__}), coba lagi {attempt}/{retries - 1}…", flush=True)
                time.sleep(5 * attempt)
    return data_dir


def load_record(name, data_dir: Path = DATA_DIR, lead: str = "MLII", download: bool = True) -> Record:
    """Baca satu rekaman MIT-BIH pada kanal `lead` beserta anotasi detak yang dipetakan ke AAMI."""
    import wfdb

    name = str(name)
    if not _complete(name, data_dir):
        if not download:
            raise FileNotFoundError(f"Rekaman {name} tidak ada di {data_dir}.")
        download_mitdb([name], data_dir)
    rec = wfdb.rdrecord(str(data_dir / name))
    # Urutan kanal tidak selalu sama (misalnya 102 dan 104), jadi cari berdasarkan nama.
    if lead not in rec.sig_name:
        raise ValueError(f"Rekaman {name} tidak punya kanal {lead} (kanal: {', '.join(rec.sig_name)}).")
    sig = rec.p_signal[:, rec.sig_name.index(lead)].astype(float)
    ann = wfdb.rdann(str(data_dir / name), "atr")
    sym = np.asarray(ann.symbol)
    smp = np.asarray(ann.sample)
    keep = np.isin(sym, list(AAMI))
    return Record(
        uid=f"mitdb-{name}", name=f"MIT-BIH {name}", signal=sig, fs=float(rec.fs),
        r_peaks=smp[keep], labels=np.array([AAMI[s] for s in sym[keep]]), lead=lead,
        description=f"MIT-BIH rekaman {name}, kanal {lead}, {rec.fs:.0f} Hz, {len(sig) / rec.fs / 60:.0f} menit. Dengan anotasi detak.",
    )


def resample_to(x: np.ndarray, fs_in: float, fs_out: float = FS_MITBIH) -> np.ndarray:
    """Sampel ulang dengan filter anti-aliasing (resample_poly)."""
    if abs(fs_in - fs_out) < 1e-9:
        return np.asarray(x, float)
    fr = Fraction(fs_out / fs_in).limit_denominator(1000)
    return resample_poly(np.asarray(x, float), fr.numerator, fr.denominator)


def _split_line(line: str) -> list[str]:
    parts = [p for p in line.replace("\t", ";").split(";")]
    if len(parts) == 1 and "," in line:
        # "0,123" dianggap angka berkoma desimal (format Excel Indonesia).
        token = line.strip()
        if token.lstrip("-").replace(",", "", 1).isdigit() and token.count(",") == 1:
            return [token.replace(",", ".")]
        parts = line.split(",")
    return [p.strip() for p in parts if p.strip()]


def load_csv(data, fs: float, name: str = "unggahan", target_fs: float = FS_MITBIH) -> Record:
    """Baca CSV satu kolom nilai sinyal. Memeriksa format, jumlah kolom, panjang, dan fs.

    Sinyal disampel ulang ke `target_fs` agar cocok dengan model yang dilatih pada 360 Hz.
    """
    raw_bytes = data if isinstance(data, bytes) else str(data).encode()
    text = raw_bytes.decode("utf-8-sig", errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Berkas kosong. Isi satu kolom nilai sinyal, satu nilai per baris.")
    if not (50 <= fs <= 2000):
        raise ValueError("Frekuensi sampling harus antara 50 dan 2000 Hz.")
    start = 0
    first = _split_line(lines[0])
    try:
        float(first[0])
    except ValueError:
        start = 1  # baris judul
    values = []
    for i, ln in enumerate(lines[start:], start=start + 1):
        cols = _split_line(ln)
        if len(cols) != 1:
            raise ValueError(f"Baris {i} berisi {len(cols)} kolom. Gunakan satu kolom nilai sinyal saja.")
        try:
            values.append(float(cols[0]))
        except ValueError:
            raise ValueError(f"Baris {i} bukan angka (\"{cols[0][:20]}\"). Periksa isi berkas.") from None
    x = np.asarray(values, float)
    dur = len(x) / fs
    if dur < 10:
        raise ValueError(f"Rekaman hanya {dur:.1f} detik. Minimal 10 detik agar detak dapat dianalisis.")
    if dur > 1800:
        raise ValueError(f"Rekaman {dur / 60:.0f} menit terlalu panjang. Maksimal 30 menit.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Berkas berisi nilai kosong atau tak hingga.")
    sig = resample_to(x, fs, target_fs)
    digest = hashlib.sha1(raw_bytes).hexdigest()[:12]
    note = " Nilai tampak dalam satuan ADC, bukan mV; klasifikasi mengasumsikan mV." if np.abs(x).max() > 20 else ""
    return Record(
        uid=f"csv-{digest}-{fs:g}", name=f"Unggahan: {name}", signal=sig, fs=float(target_fs), lead="?",
        description=f"{len(x):,} sampel pada {fs:g} Hz ({dur:.1f} s), disampel ulang ke {target_fs:g} Hz. Tanpa anotasi.{note}",
    )
