"""Segmentasi detak dan ekstraksi fitur: RR, morfologi sederhana, dan wavelet db4."""
from __future__ import annotations

import numpy as np
import pywt

PRE_S, POST_S = 0.2, 0.4          # segmen 0,2 s sebelum dan 0,4 s sesudah puncak R
SAMPLE_MS = [-50, -29, -7, 14, 36, 57, 79, 100]
WAVELET, LEVEL = "db4", 4
_COEF = ["cA4", "cD4", "cD3", "cD2", "cD1"]

FEATURES = (
    [("RR sebelum (s)", "rr"), ("RR sesudah (s)", "rr"), ("RR lokal 10 detak (s)", "rr"),
     ("Rasio RR sebelum / lokal", "rr"), ("Rasio RR sesudah / sebelum", "rr"), ("Rasio RR sebelum / rata rekaman", "rr"),
     ("Amplitudo R (mV)", "morf"), ("Lebar QRS (ms)", "morf"), ("Energi segmen", "morf"),
     ("Puncak ke puncak (mV)", "morf"), ("Nilai minimum (mV)", "morf"), ("Nilai maksimum (mV)", "morf")]
    + [(f"Sampel QRS {ms:+d} ms", "morf") for ms in SAMPLE_MS]
    + [(f"Energi {c}", "wav") for c in _COEF]
    + [(f"Simpangan baku {c}", "wav") for c in _COEF]
    # Ternormalisasi terhadap pasien: membandingkan detak dengan kebiasaan pasien itu sendiri,
    # sehingga model lebih tahan terhadap perbedaan antarpasien (pembagian DS1 → DS2).
    + [("Rasio RR sesudah / rata rekaman", "rel"), ("Rasio RR lokal / rata rekaman", "rel"),
       ("Korelasi dg. detak median pasien", "rel"), ("Korelasi QRS dg. median pasien", "rel"),
       ("Korelasi daerah P dg. median pasien", "rel"), ("Selisih daerah P thd. median (RMS)", "rel"),
       ("Rasio amplitudo R / median pasien", "rel"), ("Selisih lebar QRS thd. median (ms)", "rel")]
)
FEATURE_NAMES = [f[0] for f in FEATURES]
FEATURE_GROUPS = {g: [i for i, f in enumerate(FEATURES) if f[1] == g] for g in ("rr", "morf", "wav", "rel")}
GROUP_LABELS = {"rr": "RR", "morf": "Morfologi", "wav": "Wavelet", "rel": "Relatif pasien"}


def _rowcorr(A: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Korelasi Pearson tiap baris A dengan vektor t."""
    a = A - A.mean(axis=1, keepdims=True)
    b = t - t.mean()
    den = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum()) + 1e-12
    return (a @ b) / den


def segments(clean, peaks, fs: float) -> np.ndarray:
    """Matriks segmen (satu baris per detak), tepi sinyal diisi nilai ujung."""
    x = np.asarray(clean, float)
    B, A = int(round(PRE_S * fs)), int(round(POST_S * fs))
    idx = np.asarray(peaks)[:, None] + np.arange(-B, A)[None, :]
    return x[np.clip(idx, 0, len(x) - 1)]


def extract_features(clean, peaks, fs: float) -> np.ndarray:
    """Antarmuka yang disepakati: kembalikan matriks fitur (n_detak × 38).

    Fitur relatif pasien memakai seluruh detak rekaman, jadi panggil fungsi ini per rekaman
    (bukan per potongan pendek), seperti di train.py dan aplikasi web.
    """
    peaks = np.asarray(peaks, int)
    m = len(peaks)
    if m == 0:
        return np.zeros((0, len(FEATURES)))
    pre = np.full(m, np.nan)
    post = np.full(m, np.nan)
    pre[1:] = np.diff(peaks) / fs
    post[:-1] = np.diff(peaks) / fs
    med = np.nanmedian(pre) if m > 1 else 0.8
    pre = np.where(np.isfinite(pre), pre, med)
    post = np.where(np.isfinite(post), post, med)
    csum = np.concatenate([[0], np.cumsum(pre)])
    lo = np.maximum(0, np.arange(m) - 9)
    local = (csum[np.arange(m) + 1] - csum[lo]) / (np.arange(m) + 1 - lo)
    avg = pre.mean()

    seg = segments(clean, peaks, fs)
    seg = seg - np.median(seg, axis=1, keepdims=True)
    B = int(round(PRE_S * fs))
    lim = int(round(0.12 * fs))
    r_amp = seg[:, B]
    out = np.zeros((m, len(FEATURES)))
    for i in range(m):
        s = seg[i]
        sg = np.sign(r_amp[i]) or 1.0
        thr = 0.3 * abs(r_amp[i])
        a = b = B
        while a > B - lim and abs(s[a - 1]) > thr and np.sign(s[a - 1]) == sg:
            a -= 1
        while b < B + lim and abs(s[b + 1]) > thr and np.sign(s[b + 1]) == sg:
            b += 1
        samples = [s[int(np.clip(B + round(ms / 1000 * fs), 0, len(s) - 1))] for ms in SAMPLE_MS]
        coeffs = pywt.wavedec(s, WAVELET, level=LEVEL, mode="periodization")
        energy = [float(np.mean(c ** 2)) for c in coeffs]
        sd = [float(np.std(c, ddof=1)) for c in coeffs]
        out[i, :30] = ([pre[i], post[i], local[i], pre[i] / local[i], post[i] / pre[i], pre[i] / avg,
                        r_amp[i], (b - a + 1) / fs * 1000, float(np.mean(s ** 2)), s.max() - s.min(), s.min(), s.max()]
                       + samples + energy + sd)

    # Fitur relatif pasien: template = median semua detak rekaman (didominasi detak normal pasien itu).
    tmpl = np.median(seg, axis=0)
    p_end, q0, q1 = B - int(round(0.05 * fs)), B - int(round(0.05 * fs)), B + int(round(0.1 * fs))
    width = out[:, 7]
    med_r = np.median(np.abs(r_amp)) + 1e-9
    out[:, 30] = post / avg
    out[:, 31] = local / avg
    out[:, 32] = _rowcorr(seg, tmpl)
    out[:, 33] = _rowcorr(seg[:, q0:q1], tmpl[q0:q1])
    out[:, 34] = _rowcorr(seg[:, :p_end], tmpl[:p_end])
    out[:, 35] = np.sqrt(np.mean((seg[:, :p_end] - tmpl[:p_end]) ** 2, axis=1)) / med_r
    out[:, 36] = r_amp / (np.sign(np.median(r_amp)) * med_r)
    out[:, 37] = width - np.median(width)
    return out
