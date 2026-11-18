"""Pembangkit sinyal EKG sintetis yang meniru karakter MIT-BIH (360 Hz, mV, detak N/S/V).

Dipakai untuk demo tanpa internet, pengujian fungsi (tests/), dan sebagai fungsi tiruan
selama data asli belum siap. Angka hasil dari data sintetis TIDAK untuk dikutip di laporan.
"""
from __future__ import annotations

import numpy as np

from .data_loader import FS_MITBIH, Record

FS = FS_MITBIH

SAMPLES = {
    "SIN-01": ("Irama sinus normal", 11, dict(hr=72, pS=0.004, pV=0.004, bw=0.12, pl=0.015, emg=0.006, art=0)),
    "SIN-02": ("PVC sering", 23, dict(hr=78, pV=0.16, pS=0.01, bw=0.18, pl=0.02, emg=0.01, art=0)),
    "SIN-03": ("PAC berulang", 37, dict(hr=84, pS=0.14, pV=0.005, sPrem=0.64, bw=0.1, pl=0.02, emg=0.008, art=0)),
    "SIN-04": ("Campuran, derau berat, jala-jala 50 Hz", 41,
               dict(hr=88, pS=0.06, pV=0.07, bw=0.45, bwF=0.3, pl=0.09, plF=50, emg=0.03)),
    "SIN-05": ("Bigemini ventrikel", 53, dict(hr=64, pattern="bigemini", pS=0, pV=0.02, bw=0.15, pl=0.02, emg=0.008, art=0)),
    "SIN-06": ("Bradikardia sinus", 67, dict(hr=50, pS=0.03, pV=0.02, bw=0.2, pl=0.02, emg=0.01)),
}


def make_profile(seed: int, **overrides) -> dict:
    """Parameter acak satu 'pasien' sintetis (bentuk gelombang, irama, derau)."""
    u = np.random.default_rng(seed).random
    p = dict(
        hr=58 + u() * 34, rsa=0.02 + u() * 0.08, jit=0.01 + u() * 0.05,
        pA=0.08 + u() * 0.12, pW=0.02 + u() * 0.01, pr=0.15 + u() * 0.05,
        qA=-(0.04 + u() * 0.12), rA=0.7 + u() * 1.0, rW=0.008 + u() * 0.004, sA=-(0.08 + u() * 0.35),
        tA=0.15 + u() * 0.3, tW=0.04 + u() * 0.025, qt=0.36 + u() * 0.06,
        pS=0.03 + u() * 0.15 if u() < 0.45 else u() * 0.008,
        pV=0.03 + u() * 0.14 if u() < 0.5 else u() * 0.008,
        sPrem=0.62 + u() * 0.3, sP=(-1 if u() < 0.5 else 1) * (0.2 + u() * 0.9), sAb=1 + u() * 0.5,
        vPrem=0.55 + u() * 0.3, vW=1.5 + u() * 2.2, vA=0.7 + u() * 1.0, vPol=1 if u() < 0.6 else -1,
        vSkew=0.3 + u() * 0.5, vVar=u() * 0.3,
        bw=0.05 + u() * 0.3, bwF=0.12 + u() * 0.25, pl=u() * 0.04, plF=60, emg=0.004 + u() * 0.05,
        art=u() * 0.08 if u() < 0.4 else 0.0, dur=90.0, pattern=None,
    )
    p.update(overrides)
    return p


def _gauss(y: np.ndarray, c: float, a: float, w: float) -> None:
    i0 = max(0, int(np.floor((c - 4 * w) * FS)))
    i1 = min(len(y) - 1, int(np.ceil((c + 4 * w) * FS)))
    if i1 < i0:
        return
    t = np.arange(i0, i1 + 1) / FS - c
    y[i0:i1 + 1] += a * np.exp(-t * t / (2 * w * w))


def beat_shape(y: np.ndarray, t: float, kind: str, rr: float, p: dict, resp: float = 1.0, u: float = 0.5) -> None:
    """Tambahkan satu detak (jumlah fungsi Gauss untuk P, Q, R, S, T) pada waktu R = t."""
    rrc = float(np.clip(rr, 0.45, 1.6))
    if kind == "V":
        vv = (u - 0.5) * 2 * p["vVar"]
        A, W = p["rA"] * p["vA"] * p["vPol"] * resp * (1 + vv), p["rW"] * p["vW"] * (1 - 0.7 * vv)
        _gauss(y, t, A, W)
        _gauss(y, t + W * 2.3, -A * p["vSkew"], W * 1.25)
        _gauss(y, t - W * 1.6, -A * 0.12, W * 0.8)
        _gauss(y, t + 0.30 * np.sqrt(rrc) + 0.04, -p["vPol"] * p["tA"] * 1.7, p["tW"] * 1.45)
        return
    s = kind == "S"
    _gauss(y, t - p["pr"] + 0.01 - (0.012 if s else 0), p["pA"] * (p["sP"] if s else 1) * resp, p["pW"] * (0.85 if s else 1))
    _gauss(y, t - 0.027, p["qA"] * resp, 0.009)
    _gauss(y, t, p["rA"] * resp * (0.96 if s else 1), p["rW"] * (p["sAb"] if s else 1))
    _gauss(y, t + 0.029, p["sA"] * resp, 0.011)
    _gauss(y, t + (p["qt"] - 0.11) * np.sqrt(rrc), p["tA"] * resp, p["tW"])


def synthesize(p: dict, seed: int, uid: str = "syn", name: str = "Sintetis", description: str = "") -> Record:
    rng = np.random.default_rng(seed + 7919)
    n = int(round(p["dur"] * FS))
    clean = np.zeros(n)
    peaks, labels = [], []
    rr_n = 60.0 / p["hr"]
    ph = rng.random() * 6.28
    t, cur, last_iv = 0.35 + rng.random() * 0.5, "N", rr_n
    while t < p["dur"] - 0.45:
        resp = 1 + 0.05 * np.sin(2 * np.pi * 0.25 * t + ph)
        beat_shape(clean, t, cur, last_iv, p, resp, rng.random())
        peaks.append(int(round(t * FS)))
        labels.append(cur)
        nxt = "N"
        if p.get("pattern") == "bigemini" and int(t // 15) % 2 == 0:
            nxt = "V" if cur == "N" else "N"
        elif cur == "N":
            q = rng.random()
            nxt = "V" if q < p["pV"] else "S" if q < p["pV"] + p["pS"] else "N"
        base = rr_n * (1 + p["rsa"] * np.sin(2 * np.pi * 0.25 * t + ph) + p["jit"] * rng.standard_normal())
        if nxt == "S":
            iv = base * p["sPrem"] * (1 + 0.04 * rng.standard_normal())
        elif nxt == "V":
            iv = base * p["vPrem"] * (1 + 0.04 * rng.standard_normal())
        elif cur == "V":
            iv = max(base * 1.1, 2 * base - last_iv)  # jeda kompensasi
        elif cur == "S":
            iv = base * (1 + 0.08 * rng.random())
        else:
            iv = base
        iv = max(0.3, iv)
        last_iv, t, cur = iv, t + iv, nxt

    tt = np.arange(n) / FS
    p1, p2, p3 = rng.random(3) * 6.28
    bw = p["bw"] * (np.sin(2 * np.pi * p["bwF"] * tt + p1) + 0.45 * np.sin(2 * np.pi * p["bwF"] * 0.41 * tt + p2))
    walk = np.zeros(n)
    steps = rng.standard_normal(n) * 0.0016
    for i in range(1, n):  # random walk lambat (baseline)
        walk[i] = walk[i - 1] * 0.9993 + steps[i]
    burst = np.zeros(n)
    starts = np.flatnonzero(rng.random(n) < 0.0004)
    for s in starts:
        k = np.arange(n - s)
        burst[s:] = np.maximum(burst[s:], 0.9985 ** k)
    raw = (clean + bw + walk * p["bw"] * 4 + p["pl"] * np.sin(2 * np.pi * p["plF"] * tt + p3)
           + p["emg"] * rng.standard_normal(n) * (1 + 2.5 * burst))
    if p.get("art"):
        for s in range(int(p["dur"]) - 1):  # artefak gerakan
            if rng.random() < p["art"]:
                c = s + rng.random()
                a = (-1 if rng.random() < 0.5 else 1) * (0.4 + rng.random())
                w = 0.015 + rng.random() * 0.025
                _gauss(raw, c, a, w)
                _gauss(raw, c + 2.2 * w, -0.6 * a, w)
    return Record(uid=uid, name=name, signal=raw, fs=float(FS), r_peaks=np.asarray(peaks),
                  labels=np.asarray(labels), lead="MLII (sintetis)", clean_reference=clean,
                  description=description)


def sample_record(sid: str, duration: float = 60.0) -> Record:
    title, seed, ov = SAMPLES[sid]
    p = make_profile(seed, dur=duration, **ov)
    return synthesize(p, seed, uid=f"syn-{sid}", name=f"{sid} · {title}",
                      description=f"Sinyal sintetis meniru MIT-BIH (360 Hz, mV): {title.lower()}. Label acuan diketahui.")


def synthetic_patient(seed: int, duration: float = 90.0) -> Record:
    return synthesize(make_profile(seed, dur=duration), seed, uid=f"synpt-{seed}", name=f"Pasien sintetis {seed}")


# Pasien sintetis pengganti DS1/DS2 bila MIT-BIH belum diunduh.
SYN_DS1 = list(range(101, 123))
SYN_DS2 = list(range(201, 223))
