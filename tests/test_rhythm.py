import numpy as np

from src.rhythm import summarize

FS = 360


def _peaks(rr_s):
    return np.round(np.concatenate([[0.5], 0.5 + np.cumsum(rr_s)]) * FS).astype(int)


def test_irama_teratur_normal():
    rng = np.random.default_rng(0)
    rr = 0.8 * (1 + 0.01 * rng.standard_normal(300))
    s = summarize(_peaks(rr), np.full(301, "N"), FS)
    assert s["verdict"][1] == "ok"
    assert s["regularity"][1] == "ok" and s["rate"][1] == "ok"


def test_fibrilasi_atrium_terdeteksi():
    rng = np.random.default_rng(1)
    rr = rng.uniform(0.45, 1.1, 300)  # RR acak tanpa pola
    s = summarize(_peaks(rr), np.full(301, "N"), FS)
    assert s["irregular_frac"] >= 0.5
    assert "fibrilasi atrium" in s["verdict"][0]


def test_bigemini_ventrikel_bukan_fibrilasi():
    rr = np.tile([0.5, 1.1], 150)  # pendek (detak V) lalu jeda kompensasi
    pred = np.array(["N", "V"] * 150 + ["N"])
    s = summarize(_peaks(rr), pred, FS)
    assert "fibrilasi" not in s["verdict"][0]
    assert "detak prematur sering" in s["verdict"][0]


def test_bradikardia():
    s = summarize(_peaks(np.full(100, 1.25)), np.full(101, "N"), FS)  # 48 bpm
    assert "bradikardia" in s["verdict"][0]


def test_rekaman_pendek():
    s = summarize(_peaks(np.full(8, 0.8)), np.full(9, "N"), FS)
    assert s["irregular_frac"] is None and s["regularity"][1] == "na"
