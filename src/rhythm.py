"""Ringkasan irama berbasis aturan: detak prematur, kecepatan (HR), dan keteraturan interval RR.

Aturan sengaja dibuat sederhana dan terbuka agar mudah dijelaskan; ini bukan diagnosis medis.
"""
from __future__ import annotations

import numpy as np

WINDOW = 30             # jumlah interval RR per jendela penilaian keteraturan
IRREG_THR = 0.08        # median |ΔRR| / median RR di atas ini → jendela dianggap tidak teratur
AF_FRAC = 0.5           # ≥ 50% jendela tidak teratur → curiga fibrilasi atrium
PERIOD_FRAC = 0.2       # 20–50% → ada periode tidak teratur
PREMATURE_PCT = 1.0     # ≥ 1% detak S+V dianggap sering
BRADY, TACHY = 60, 100  # batas HR (bpm)


def irregularity(peaks, pred, fs: float):
    """Nilai ketidakteraturan tiap jendela 30 RR, setelah RR di sekitar detak V dibuang.

    Detak V dikeluarkan agar bigemini/trigemini ventrikel tidak terbaca sebagai fibrilasi atrium.
    Detak S tidak dikeluarkan, karena pada fibrilasi atrium model sering melabeli detak tak teratur sebagai S.
    """
    peaks = np.asarray(peaks)
    rr = np.diff(peaks) / fs
    ok = np.ones(len(rr), bool)
    for j in np.flatnonzero(np.asarray(pred) == "V"):
        ok[max(j - 1, 0)] = False
        if j < len(rr):
            ok[j] = False
    n_win = max(1, len(rr) // WINDOW)
    size = len(rr) if len(rr) < WINDOW else WINDOW
    vals = []
    for w in range(n_win):
        seg = rr[w * size:(w + 1) * size][ok[w * size:(w + 1) * size]]
        if len(seg) < 12:
            continue
        vals.append(float(np.median(np.abs(np.diff(seg))) / np.median(seg)))
    return np.array(vals)


def summarize(peaks, pred, fs: float) -> dict:
    """Kembalikan ringkasan irama: angka-angka dan kalimat kesimpulan."""
    pred = np.asarray(pred)
    n = len(pred)
    cnt = {k: int(np.sum(pred == k)) for k in ("N", "S", "V")}
    rr = np.diff(np.asarray(peaks)) / fs
    hr = float(60 / rr.mean()) if len(rr) else float("nan")  # sama dengan HR rata-rata di halaman Deteksi R
    irr = irregularity(peaks, pred, fs)
    frac = float(np.mean(irr > IRREG_THR)) if len(irr) else None

    if not np.isfinite(hr):
        rate = ("tidak diketahui", "warn")
    elif hr < BRADY:
        rate = ("bradikardia (lebih lambat dari 60 bpm)", "warn")
    elif hr > TACHY:
        rate = ("takikardia (lebih cepat dari 100 bpm)", "warn")
    else:
        rate = ("normal (60–100 bpm)", "ok")

    if frac is None:
        reg = ("rekaman terlalu pendek untuk menilai keteraturan (perlu minimal 13 detak)", "na")
    elif frac >= AF_FRAC:
        reg = (f"sangat tidak teratur pada {frac:.0%} rekaman: curiga fibrilasi atrium", "warn")
    elif frac >= PERIOD_FRAC:
        reg = (f"ada periode tidak teratur ({frac:.0%} rekaman)", "warn")
    else:
        reg = ("teratur", "ok")

    prem = cnt["S"] + cnt["V"]
    pct = 100 * prem / n if n else 0.0
    parts = []
    for k, nama in (("S", "supraventrikular"), ("V", "ventrikular")):
        if cnt[k]:
            jumlah = f"{cnt[k]:,}".replace(",", ".")
            p = 100 * cnt[k] / n
            persen = "<0,1" if p < 0.05 else f"{p:.1f}".replace(".", ",")
            parts.append(f"{jumlah} detak {k} ({nama}, {persen}%)")
    if not prem:
        premature = ("tidak ditemukan", "ok")
    elif pct >= PREMATURE_PCT:
        premature = ("sering: " + ", ".join(parts), "warn")
    else:
        premature = ("sesekali: " + ", ".join(parts) + "; jumlah sedikit seperti ini juga bisa terjadi pada orang sehat", "mild")

    signs = []
    if reg[1] == "warn" and frac is not None and frac >= AF_FRAC:
        signs.append("irama sangat tidak teratur (curiga fibrilasi atrium)")
    elif reg[1] == "warn":
        signs.append("periode irama tidak teratur")
    if premature[1] == "warn":
        signs.append("detak prematur sering")
    if rate[1] == "warn":
        signs.append(rate[0].split(" (")[0])
    if signs:
        verdict = ("Ada tanda aritmia: " + ", ".join(signs) + ".", "warn")
    elif premature[1] == "mild":
        verdict = ("Irama dasar normal, dengan detak prematur sesekali.", "mild")
    else:
        verdict = ("Tidak ada tanda aritmia.", "ok")

    note = None
    if frac is not None and frac >= AF_FRAC and cnt["S"]:
        note = ("Pada irama yang sangat tidak teratur, label S dari model belum tentu detak prematur sungguhan: "
                "model hanya dilatih untuk N, S, dan V, bukan fibrilasi atrium.")
    return {"n": n, "count": cnt, "premature_pct": pct, "hr": hr, "irregular_frac": frac,
            "rate": rate, "regularity": reg, "premature": premature, "verdict": verdict, "note": note}
