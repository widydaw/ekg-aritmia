"""Deteksi puncak R dengan algoritma Pan-Tompkins (1985) yang ditulis sendiri."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


@dataclass
class PanTompkinsResult:
    peaks: np.ndarray           # indeks puncak R pada sinyal masukan
    bandpassed: np.ndarray      # (a) band-pass 5–15 Hz
    derivative: np.ndarray      # (b) turunan
    squared: np.ndarray         # (c) kuadrat
    integrated: np.ndarray      # (d) integrasi jendela bergerak
    threshold_idx: np.ndarray   # (e) posisi tiap pembaruan ambang
    threshold_val: np.ndarray   #     nilai ambang saat itu


def pan_tompkins(x, fs: float, refine_on=None) -> PanTompkinsResult:
    x = np.asarray(x, float)
    n = len(x)
    ref_on = x if refine_on is None else np.asarray(refine_on, float)

    # (a) band-pass sekitar 5–15 Hz menonjolkan energi QRS
    bp = sosfiltfilt(butter(2, [5, 15], btype="bandpass", fs=fs, output="sos"), x)
    # (b) turunan lima titik menekankan kemiringan QRS
    d = np.zeros(n)
    d[2:-2] = (2 * bp[3:-1] + bp[4:] - bp[:-4] - 2 * bp[1:-3]) * fs / 8
    # (c) pengkuadratan
    sq = d ** 2
    # (d) integrasi jendela bergerak 150 ms (berpusat agar tidak ada tunda)
    w = int(round(0.15 * fs))
    mwi = np.convolve(sq, np.ones(w) / w, mode="same")

    # Kandidat: maksimum lokal yang terpisah minimal 200 ms (periode refrakter)
    refr = int(round(0.2 * fs))
    cand, _ = find_peaks(mwi, distance=refr)

    # (e) ambang adaptif: dua detik pertama sebagai fase belajar
    lw = min(n, int(2 * fs))
    spki, npki = mwi[:lw].max() / 3, mwi[:lw].mean() / 2
    thr = npki + 0.25 * (spki - npki)
    hw = int(round(0.075 * fs))
    qrs: list[int] = []
    rrs: list[int] = []
    last_slope = 0.0
    th_i, th_v = [], []

    def slope(i):
        return np.abs(d[max(0, i - hw):i + hw + 1]).max()

    def push(c):
        if qrs:
            rrs.append(c - qrs[-1])
            del rrs[:-8]
        qrs.append(int(c))

    for c in cand:
        pk = mwi[c]
        th_i.append(c)
        th_v.append(thr)
        if pk > thr:
            last = qrs[-1] if qrs else -10 ** 9
            if c - last < refr:
                continue
            sl = slope(c)
            # Pemeriksaan gelombang T: kemiringan jauh lebih kecil dari QRS sebelumnya
            if c - last < 0.36 * fs and sl < 0.5 * last_slope:
                npki = 0.125 * pk + 0.875 * npki
            else:
                rr_avg = np.mean(rrs) if rrs else 0
                # Pencarian mundur bila jeda > 166% RR rata-rata
                if rr_avg and last > 0 and c - last > 1.66 * rr_avg:
                    lo = np.searchsorted(cand, last + refr, side="right")
                    hi = np.searchsorted(cand, c - refr, side="left")
                    seg = cand[lo:hi]
                    if len(seg):
                        b = seg[np.argmax(mwi[seg])]
                        if mwi[b] > 0.5 * thr:
                            push(b)
                            spki = 0.25 * mwi[b] + 0.75 * spki
                push(c)
                last_slope = sl
                spki = 0.125 * pk + 0.875 * spki
        else:
            npki = 0.125 * pk + 0.875 * npki
        thr = npki + 0.25 * (spki - npki)

    # Perhalus posisi R pada sinyal bersih (±100 ms), mendukung QRS berpolaritas negatif
    rw = int(round(0.1 * fs))
    peaks: list[int] = []
    for q in qrs:
        a, b = max(0, q - rw), min(n, q + rw + 1)
        seg = ref_on[a:b]
        m = seg.mean()
        imax, imin = int(np.argmax(seg)), int(np.argmin(seg))
        p = a + (imin if (m - seg[imin]) > 1.4 * (seg[imax] - m) else imax)
        if peaks and p - peaks[-1] < refr:
            if abs(ref_on[p]) > abs(ref_on[peaks[-1]]):
                peaks[-1] = p
        else:
            peaks.append(p)
    return PanTompkinsResult(np.asarray(peaks, int), bp, d, sq, mwi, np.asarray(th_i), np.asarray(th_v))


def detect_r_peaks(clean, fs: float) -> np.ndarray:
    """Antarmuka yang disepakati: sinyal bersih → larik indeks puncak R."""
    return pan_tompkins(clean, fs, refine_on=clean).peaks


def match_beats(detected, reference, fs: float, tol_s: float = 0.15):
    """Pasangkan deteksi dengan anotasi dalam toleransi waktu (AAMI: 150 ms).

    Mengembalikan (pasangan [(i_det, j_ref)], TP, FP, FN).
    """
    tol = int(round(tol_s * fs))
    det, ref = np.asarray(detected), np.asarray(reference)
    i = j = 0
    pairs = []
    while i < len(det) and j < len(ref):
        dlt = det[i] - ref[j]
        if abs(dlt) <= tol:
            pairs.append((i, j))
            i += 1
            j += 1
        elif dlt < 0:
            i += 1
        else:
            j += 1
    tp = len(pairs)
    return pairs, tp, len(det) - tp, len(ref) - tp


def detection_metrics(detected, reference, fs: float, tol_s: float = 0.15) -> dict:
    _, tp, fp, fn = match_beats(detected, reference, fs, tol_s)
    return {"TP": tp, "FP": fp, "FN": fn,
            "sensitivitas": tp / (tp + fn) if tp + fn else float("nan"),
            "presisi": tp / (tp + fp) if tp + fp else float("nan")}


def heart_rate(peaks, fs: float) -> dict:
    """HR = 60 / RR, serta HRV sederhana (SDNN, RMSSD) dalam ms."""
    rr = np.diff(np.asarray(peaks)) / fs * 1000
    if len(rr) < 2:
        return {"rr_ms": rr, "hr": float("nan"), "sdnn": float("nan"), "rmssd": float("nan")}
    return {"rr_ms": rr, "hr": 60000 / rr.mean(), "sdnn": float(rr.std(ddof=1)),
            "rmssd": float(np.sqrt(np.mean(np.diff(rr) ** 2)))}
