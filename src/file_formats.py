"""Membaca rekaman EKG unggahan dalam beberapa format standar.

Format yang didukung:
- WFDB (PhysioNet): .hea + .dat (atau .mat), opsional anotasi .atr / .qrs
- EDF / EDF+ (European Data Format): .edf
- HL7 aECG (annotated ECG, XML standar FDA): .xml
- MATLAB: .mat (tanpa .hea, frekuensi sampling diisi manual)
- CSV / TXT: satu kolom atau beberapa kolom (satu kolom per sadapan), boleh dengan judul

Semua sinyal dikembalikan dalam mV. Rekaman dipilih per sadapan lalu disampel ulang
ke 360 Hz agar diproses sama seperti MIT-BIH.
"""
from __future__ import annotations

import hashlib
import io
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .data_loader import AAMI, FS_MITBIH, Record, resample_to

FORMATS = {
    "WFDB (PhysioNet)": "Unggah .hea bersama .dat (atau .mat). Tambahkan .atr bila ada anotasi detak.",
    "EDF / EDF+": "Satu berkas .edf. Frekuensi sampling dan satuan dibaca dari header.",
    "HL7 aECG (XML)": "Satu berkas .xml format annotated ECG (standar FDA).",
    "MATLAB (.mat)": "Satu berkas .mat berisi matriks sinyal. Isi frekuensi sampling secara manual.",
    "CSV / TXT": "Satu kolom per sadapan, boleh dengan baris judul. Kolom waktu dikenali otomatis.",
}
EXTENSIONS = ["hea", "dat", "atr", "qrs", "edf", "xml", "mat", "csv", "txt"]
MIN_S, MAX_S = 10.0, 1800.0


@dataclass
class Upload:
    name: str
    fmt: str
    channels: dict = field(default_factory=dict)   # nama sadapan -> (sinyal mV, fs)
    r_peaks: np.ndarray | None = None              # indeks sampel anotasi (pada fs kanal)
    labels: np.ndarray | None = None
    notes: list = field(default_factory=list)
    digest: str = ""


# ---------------------------------------------------------------- satuan
def _to_mv(x: np.ndarray, unit: str) -> np.ndarray:
    u = (unit or "").strip().lower().replace("μ", "u").replace("µ", "u")
    if u in ("uv", "microvolt", "microvolts"):
        return x / 1000.0
    if u in ("v", "volt", "volts"):
        return x * 1000.0
    if u in ("nv",):
        return x / 1e6
    return x


def _guess_mv(x: np.ndarray, notes: list, label: str) -> np.ndarray:
    """Tanpa satuan: puncak lebih dari 20 dianggap µV (EKG normal hanya beberapa mV)."""
    if np.nanmax(np.abs(x)) > 20:
        notes.append(f"{label}: nilai tampak dalam µV, dibagi 1000 menjadi mV.")
        return x / 1000.0
    return x


# ---------------------------------------------------------------- WFDB
def _read_wfdb(files: dict) -> Upload:
    import wfdb

    heas = [n for n in files if n.lower().endswith(".hea")]
    if len(heas) != 1:
        raise ValueError("Unggah tepat satu berkas .hea untuk format WFDB.")
    hea = heas[0]
    base = hea[:-4]
    with tempfile.TemporaryDirectory() as td:
        for n, b in files.items():
            Path(td, n).write_bytes(b)
        text = files[hea].decode("utf-8", errors="replace")
        needed = {ln.split()[0] for ln in text.splitlines()[1:] if ln.strip() and not ln.startswith("#")}
        missing = [n for n in needed if n not in files and not n.startswith("~")]
        if missing:
            raise ValueError(f"Berkas data {', '.join(sorted(missing))} yang disebut di {hea} belum diunggah. "
                             "Unggah .hea bersama berkas datanya sekaligus.")
        rec = wfdb.rdrecord(str(Path(td, base)))
        up = Upload(name=base, fmt="WFDB")
        units = rec.units or [""] * rec.n_sig
        for i, nm in enumerate(rec.sig_name):
            up.channels[str(nm)] = (_to_mv(rec.p_signal[:, i].astype(float), units[i]), float(rec.fs))
        for ext in ("atr", "qrs"):
            if f"{base}.{ext}" in files:
                ann = wfdb.rdann(str(Path(td, base)), ext)
                sym, smp = np.asarray(ann.symbol), np.asarray(ann.sample)
                keep = np.isin(sym, list(AAMI))
                up.r_peaks, up.labels = smp[keep], np.array([AAMI[s] for s in sym[keep]])
                up.notes.append(f"Anotasi .{ext} dimuat: {int(keep.sum()):,} detak.")
                break
    return up


# ---------------------------------------------------------------- EDF / EDF+
def _read_edf(name: str, b: bytes) -> Upload:
    def txt(a, n):
        return b[a:a + n].decode("latin-1").strip()

    if len(b) < 256 or not txt(0, 8).startswith("0"):
        raise ValueError("Berkas .edf tidak dikenali: header EDF tidak valid.")
    hdr_bytes = int(txt(184, 8))
    n_rec = int(txt(236, 8))
    dur = float(txt(244, 8))
    ns = int(txt(252, 4))
    p = 256

    def field_(width):
        nonlocal p
        vals = [txt(p + i * width, width) for i in range(ns)]
        p += ns * width
        return vals

    labels = field_(16); field_(80); dims = field_(8)
    pmin = [float(v) for v in field_(8)]; pmax = [float(v) for v in field_(8)]
    dmin = [float(v) for v in field_(8)]; dmax = [float(v) for v in field_(8)]
    field_(80); nsamp = [int(v) for v in field_(8)]; field_(32)
    rec_len = sum(nsamp)
    if n_rec < 0:
        n_rec = (len(b) - hdr_bytes) // (2 * rec_len)
    raw = np.frombuffer(b, dtype="<i2", count=n_rec * rec_len, offset=hdr_bytes).reshape(n_rec, rec_len)
    up = Upload(name=Path(name).stem, fmt="EDF+" if txt(192, 44).startswith("EDF+") else "EDF")
    off = 0
    for i in range(ns):
        seg = raw[:, off:off + nsamp[i]].reshape(-1).astype(float)
        off += nsamp[i]
        if "annotation" in labels[i].lower():
            continue
        gain = (pmax[i] - pmin[i]) / ((dmax[i] - dmin[i]) or 1)
        phys = (seg - dmin[i]) * gain + pmin[i]
        lab = labels[i] or f"Kanal {i + 1}"
        up.channels[lab] = (_to_mv(phys, dims[i]), nsamp[i] / dur)
    if not up.channels:
        raise ValueError("Berkas .edf tidak berisi kanal sinyal.")
    return up


# ---------------------------------------------------------------- HL7 aECG
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _read_hl7(name: str, b: bytes) -> Upload:
    try:
        root = ET.fromstring(b)
    except ET.ParseError as e:
        raise ValueError(f"Berkas XML tidak bisa dibaca: {e}.") from None
    up = Upload(name=Path(name).stem, fmt="HL7 aECG")
    for sset in (el for el in root.iter() if _local(el.tag) == "sequenceSet"):
        fs = None
        leads = []
        for seq in (el for el in sset if _local(el.tag) == "component"):
            s = next((c for c in seq if _local(c.tag) == "sequence"), None)
            if s is None:
                continue
            code = next((c for c in s if _local(c.tag) == "code"), None)
            val = next((c for c in s if _local(c.tag) == "value"), None)
            if code is None or val is None:
                continue
            cname = code.get("code", "")
            if "TIME" in cname.upper():
                inc = next((c for c in val if _local(c.tag) == "increment"), None)
                if inc is not None:
                    step = float(inc.get("value"))
                    step = step / 1000.0 if inc.get("unit", "s") == "ms" else step
                    fs = 1.0 / step
                continue
            dig = next((c for c in val if _local(c.tag) == "digits"), None)
            if dig is None or not (dig.text or "").strip():
                continue
            origin = next((c for c in val if _local(c.tag) == "origin"), None)
            scale = next((c for c in val if _local(c.tag) == "scale"), None)
            o = float(origin.get("value", 0)) if origin is not None else 0.0
            sc = float(scale.get("value", 1)) if scale is not None else 1.0
            unit = scale.get("unit", "uV") if scale is not None else "uV"
            x = o + sc * np.array(dig.text.split(), dtype=float)
            leads.append((cname.replace("MDC_ECG_LEAD_", "") or f"Kanal {len(leads) + 1}", _to_mv(x, unit)))
        if fs and leads:
            for nm, x in leads:
                up.channels[nm] = (x, fs)
            return up
    raise ValueError("Tidak ditemukan gelombang EKG (sequenceSet dengan digits) di berkas XML ini. "
                     "Pastikan formatnya HL7 aECG.")


# ---------------------------------------------------------------- MATLAB
def _read_mat(name: str, b: bytes, fs: float) -> Upload:
    from scipy.io import loadmat

    try:
        m = loadmat(io.BytesIO(b))
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Berkas .mat tidak bisa dibaca: {e}. Simpan ulang dengan versi -v7 atau lebih lama.") from None
    arrays = {k: np.asarray(v) for k, v in m.items() if not k.startswith("__")
              and isinstance(v, np.ndarray) and v.dtype.kind in "iuf" and v.size > 1}
    if not arrays:
        raise ValueError("Berkas .mat tidak berisi matriks angka.")
    key = "val" if "val" in arrays else max(arrays, key=lambda k: arrays[k].size)
    a = np.atleast_2d(arrays[key].astype(float))
    if a.shape[0] > a.shape[1]:
        a = a.T  # baris = sadapan
    up = Upload(name=Path(name).stem, fmt="MATLAB")
    up.notes.append(f"Variabel '{key}' dipakai ({a.shape[0]} sadapan × {a.shape[1]:,} sampel).")
    a = _guess_mv(a, up.notes, "Semua kanal")
    for i, row in enumerate(a):
        up.channels[f"Kanal {i + 1}"] = (row, fs)
    return up


# ---------------------------------------------------------------- CSV / TXT
_TIME_NAMES = re.compile(r"^(time|times|waktu|t|sec|second|seconds|detik|ms|timestamp|sample|samples|index|no)\b",
                         re.IGNORECASE)


def _split(line: str, delim: str | None) -> list[str]:
    if delim:
        return [c.strip() for c in line.split(delim)]
    token = line.strip()
    if "," in token and re.fullmatch(r"-?\d+,\d+", token):
        return [token.replace(",", ".")]  # koma desimal ala Excel Indonesia
    return [c.strip() for c in token.split(",")] if "," in token else [token]


def _num(s: str) -> float:
    return float(s.replace(",", ".")) if re.fullmatch(r"-?\d+,\d+", s) else float(s)


def _read_csv(name: str, b: bytes, fs: float) -> Upload:
    text = b.decode("utf-8-sig", errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Berkas kosong. Isi satu kolom per sadapan, satu sampel per baris.")
    sample = "\n".join(lines[:5])
    delim = ";" if ";" in sample else "\t" if "\t" in sample else None
    first = _split(lines[0], delim)
    try:
        [_num(c) for c in first]
        header, start = None, 0
    except ValueError:
        header, start = first, 1
    ncol = len(first)
    rows = []
    for i, ln in enumerate(lines[start:], start=start + 1):
        cols = _split(ln, delim)
        if len(cols) != ncol:
            raise ValueError(f"Baris {i} berisi {len(cols)} kolom, padahal baris pertama {ncol} kolom. "
                             "Pastikan setiap baris punya jumlah kolom yang sama.")
        try:
            rows.append([_num(c) for c in cols])
        except ValueError:
            bad = next(c for c in cols if not re.fullmatch(r"-?[\d.,eE+-]+", c))
            raise ValueError(f"Baris {i} bukan angka (\"{bad[:20]}\"). Periksa isi berkas.") from None
    a = np.asarray(rows, float)
    names = header or [f"Kanal {j + 1}" if ncol > 1 else "Sinyal" for j in range(ncol)]
    up = Upload(name=Path(name).stem, fmt="CSV")
    keep = list(range(ncol))
    if ncol > 1:
        t = a[:, 0]
        d = np.diff(t)
        is_time = (header and _TIME_NAMES.match(header[0] or "")) or (len(d) and np.all(d > 0) and np.allclose(d, d[0], rtol=1e-3))
        if is_time:
            keep = keep[1:]
            step = float(np.median(d))
            unit_ms = header and re.search(r"ms|milli", header[0], re.IGNORECASE)
            if step > 0 and not (header and re.search(r"sample|index|no", header[0], re.IGNORECASE)):
                fs_t = 1000.0 / step if unit_ms else (1.0 / step if step < 1 else None)
                if fs_t and 50 <= fs_t <= 2000:
                    fs = fs_t
                    up.notes.append(f"Frekuensi sampling {fs:g} Hz dihitung dari kolom waktu.")
            up.notes.append(f"Kolom pertama '{names[0]}' dianggap waktu dan tidak ditampilkan.")
    for j in keep:
        up.channels[names[j] or f"Kanal {j + 1}"] = (a[:, j], fs)
    big = max(np.nanmax(np.abs(up.channels[k][0])) for k in up.channels)
    if big > 20:
        up.notes.append("Nilai tampak dalam µV (puncak lebih dari 20), dibagi 1000 menjadi mV.")
        up.channels = {k: (v / 1000.0, f) for k, (v, f) in up.channels.items()}
    return up


# ---------------------------------------------------------------- pintu utama
def read_upload(files: list[tuple[str, bytes]], fs_hint: float = 360.0) -> Upload:
    """Kenali format dari ekstensi berkas lalu baca semua sadapannya."""
    if not files:
        raise ValueError("Belum ada berkas yang diunggah.")
    if not (50 <= fs_hint <= 2000):
        raise ValueError("Frekuensi sampling harus antara 50 dan 2000 Hz.")
    by = {n: b for n, b in files}
    exts = {Path(n).suffix.lower().lstrip(".") for n in by}
    unknown = exts - set(EXTENSIONS)
    if unknown:
        raise ValueError(f"Ekstensi .{', .'.join(sorted(unknown))} belum didukung. Format yang bisa dipakai: "
                         + ", ".join(FORMATS) + ".")
    if "hea" in exts:
        up = _read_wfdb(by)
    elif exts & {"dat", "atr", "qrs"}:
        raise ValueError("Berkas WFDB perlu berkas .hea. Unggah .hea dan .dat secara bersamaan.")
    else:
        if len(by) > 1:
            raise ValueError("Untuk format selain WFDB, unggah satu berkas saja.")
        n, b = next(iter(by.items()))
        ext = Path(n).suffix.lower()
        if ext == ".edf":
            up = _read_edf(n, b)
        elif ext == ".xml":
            up = _read_hl7(n, b)
        elif ext == ".mat":
            up = _read_mat(n, b, fs_hint)
        else:
            up = _read_csv(n, b, fs_hint)
    h = hashlib.sha1()
    for n in sorted(by):
        h.update(n.encode()); h.update(by[n])
    up.digest = h.hexdigest()[:12]
    return up


def default_channel(up: Upload) -> str:
    """Utamakan sadapan II / MLII seperti MIT-BIH."""
    names = list(up.channels)
    for want in ("MLII", "II", "ECG", "ECG1"):
        for n in names:
            if n.strip().upper() == want:
                return n
    for n in names:
        if "II" in n.upper() and "III" not in n.upper():
            return n
    return names[0]


def to_record(up: Upload, channel: str, target_fs: float = FS_MITBIH) -> Record:
    x, fs = up.channels[channel]
    x = np.asarray(x, float)
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=float(np.nanmedian(x)))
    dur = len(x) / fs
    notes = list(up.notes)
    if dur < MIN_S:
        raise ValueError(f"Rekaman hanya {dur:.1f} detik. Minimal {MIN_S:.0f} detik agar detak dapat dianalisis.")
    if dur > MAX_S:
        x = x[: int(MAX_S * fs)]
        notes.append(f"Rekaman {dur / 60:.0f} menit dipotong menjadi 30 menit pertama.")
    sig = resample_to(x, fs, target_fs)
    peaks = labels = None
    if up.r_peaks is not None:
        p = np.round(up.r_peaks * target_fs / fs).astype(int)
        m = p < len(sig)
        peaks, labels = p[m], up.labels[m]
    desc = (f"{up.fmt}, sadapan {channel}, {fs:g} Hz, {len(x) / fs:.1f} s"
            + (f", disampel ulang ke {target_fs:g} Hz" if abs(fs - target_fs) > 1e-9 else "")
            + (". Dengan anotasi detak." if peaks is not None else ". Tanpa anotasi.")
            + ("".join(" " + n for n in notes)))
    return Record(uid=f"up-{up.digest}-{channel}-{fs:g}", name=f"Unggahan: {up.name}", signal=sig, fs=float(target_fs),
                  r_peaks=peaks, labels=labels, lead=channel, description=desc)
