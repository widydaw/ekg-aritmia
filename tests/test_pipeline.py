"""Pengujian fungsi tanpa membuka web: jalankan dengan  pytest"""
import numpy as np
import pytest

from src.data_loader import load_csv
from src.evaluation import evaluate
from src.features import FEATURE_NAMES, extract_features
from src.preprocessing import STANDARD, filter_metrics, preprocess
from src.qrs_detection import detection_metrics, detect_r_peaks
from src.synthetic import sample_record

FS = 360


@pytest.fixture(scope="module")
def rec():
    return sample_record("SIN-04", duration=40)  # derau berat, jala-jala 50 Hz


def test_preprocess_keeps_length_and_improves_snr(rec):
    clean = preprocess(rec.signal, rec.fs, **STANDARD)
    assert clean.shape == rec.signal.shape
    m = filter_metrics(rec.signal, clean, rec.fs, rec.clean_reference, rec.r_peaks)
    assert m["snr_after"] > m["snr_before"] + 5
    assert m["baseline_db"] > 10
    assert m["qrs_corr"] > 0.9


def test_r_peak_detection_matches_annotations(rec):
    clean = preprocess(rec.signal, rec.fs, **STANDARD)
    m = detection_metrics(detect_r_peaks(clean, rec.fs), rec.r_peaks, rec.fs)
    assert m["sensitivitas"] > 0.97 and m["presisi"] > 0.95


def test_features_shape_and_finite(rec):
    clean = preprocess(rec.signal, rec.fs, **STANDARD)
    F = extract_features(clean, rec.r_peaks, rec.fs)
    assert F.shape == (len(rec.r_peaks), len(FEATURE_NAMES)) == (len(rec.r_peaks), 38)
    assert np.isfinite(F).all()


def test_evaluate_matches_example_from_design():
    # Contoh rancangan untuk satu kelas: TP 90, FN 10, FP 50, TN 950.
    y_true = ["V"] * 100 + ["N"] * 1000
    y_pred = ["V"] * 90 + ["N"] * 10 + ["V"] * 50 + ["N"] * 950
    ev = evaluate(y_true, y_pred, labels=["N", "V"])
    v = ev["per_class"].set_index("kelas").loc["V"]
    assert ev["accuracy"] == pytest.approx(0.945, abs=1e-3)
    assert v["sensitivitas"] == pytest.approx(0.90)
    assert v["presisi"] == pytest.approx(0.643, abs=1e-3)


def test_csv_validation_and_resampling():
    sig = np.sin(np.arange(250 * 12) / 250 * 2 * np.pi)
    rec = load_csv("sinyal\n" + "\n".join(f"{v:.4f}" for v in sig), fs=250)
    assert rec.fs == FS and abs(rec.duration - 12) < 0.05
    with pytest.raises(ValueError, match="kolom"):
        load_csv("1;2\n3;4\n" * 2000, fs=360)
    with pytest.raises(ValueError, match="Minimal 10 detik"):
        load_csv("\n".join(["0.1"] * 100), fs=360)
    with pytest.raises(ValueError, match="bukan angka"):
        load_csv("0.1\nabc\n" * 3000, fs=360)
