"""Filter dan penghilangan derau EKG, serta ukuran perbaikan sinyal."""
from __future__ import annotations

import numpy as np
from scipy import signal as ss
from scipy.ndimage import median_filter

MODES = {
    "bandpass": "Band-pass Butterworth",
    "bandpass_notch": "Band-pass + notch",
    "median": "Filter median + low-pass",
}
STANDARD = dict(mode="bandpass", low=0.5, high=40.0, order=4, notch=None)


def _odd(v: float) -> int:
    return int(round(v)) | 1


def preprocess(x, fs: float, mode: str = "bandpass", low: float = 0.5, high: float = 40.0,
               order: int = 4, notch: float | None = None) -> np.ndarray:
    """Kembalikan sinyal bersih.

    - bandpass: Butterworth band-pass `low`–`high` Hz, diterapkan dua arah (sosfiltfilt)
      sehingga fase tidak bergeser dan bentuk QRS terjaga.
    - bandpass_notch: seperti di atas, ditambah notch `notch` Hz (50 atau 60).
    - median: baseline diperkirakan dengan median 200 ms lalu 600 ms, dikurangkan,
      lalu low-pass `high` Hz.
    """
    x = np.asarray(x, float)
    high = min(high, 0.45 * fs)
    if mode == "median":
        base = median_filter(median_filter(x, _odd(0.2 * fs), mode="nearest"), _odd(0.6 * fs), mode="nearest")
        sos = ss.butter(order, high, btype="low", fs=fs, output="sos")
        return ss.sosfiltfilt(sos, x - base)
    sos = ss.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    y = ss.sosfiltfilt(sos, x)
    if mode == "bandpass_notch" and notch:
        b, a = ss.iirnotch(notch, Q=30, fs=fs)
        y = ss.filtfilt(b, a, y)
    return y


def spectrum(x, fs: float, nperseg: int = 4096):
    """Rapat spektral daya metode Welch (mV²/Hz)."""
    x = np.asarray(x, float)
    return ss.welch(x, fs=fs, nperseg=min(nperseg, len(x)))


def band_power(f, P, lo: float, hi: float) -> float:
    m = (f >= lo) & (f < hi)
    return float(np.sum(P[m]) * (f[1] - f[0]))


def snr_db(reference, x) -> float:
    """SNR terhadap sinyal acuan bersih (hanya bila acuan tersedia)."""
    r = np.asarray(reference) - np.mean(reference)
    y = np.asarray(x) - np.mean(x)
    return float(10 * np.log10(np.sum(r ** 2) / np.sum((y - r) ** 2)))


def filter_metrics(raw, clean, fs: float, reference=None, r_peaks=None) -> dict:
    """Penurunan daya (dB) pada pita derau, ditambah SNR dan kemiripan QRS bila ada acuan."""
    f, Pr = spectrum(raw, fs)
    _, Pc = spectrum(clean, fs)

    def red(lo, hi):
        return 10 * np.log10(band_power(f, Pr, lo, hi) / max(band_power(f, Pc, lo, hi), 1e-14))

    mains_r = band_power(f, Pr, 49, 51) + band_power(f, Pr, 59, 61)
    mains_c = band_power(f, Pc, 49, 51) + band_power(f, Pc, 59, 61)
    out = {
        "baseline_db": red(0, 0.5),
        "mains_db": 10 * np.log10(mains_r / max(mains_c, 1e-14)),
        "hf_db": red(40, fs / 2),
    }
    if reference is not None:
        out["snr_before"] = snr_db(reference, raw)
        out["snr_after"] = snr_db(reference, clean)
        if r_peaks is not None and len(r_peaks):
            hw = int(0.05 * fs)
            idx = np.concatenate([np.arange(p - hw, p + hw + 1) for p in r_peaks])
            idx = idx[(idx >= 0) & (idx < len(clean))]
            out["qrs_corr"] = float(np.corrcoef(np.asarray(reference)[idx], np.asarray(clean)[idx])[0, 1])
    return out
