"""Latih model pada DS1, uji pada DS2, lalu simpan model dan hasil evaluasi ke folder models/.

Pemakaian:
    python train.py               # MIT-BIH, diunduh otomatis dari PhysioNet ke data/mitdb (±100 MB)
    python train.py --tune        # + setel parameter Random Forest (validasi silang per rekaman)
    python train.py --synthetic   # data sintetis: cepat, tanpa internet, HANYA untuk uji coba

Fitur latih diambil pada posisi R dari anotasi agar kinerja klasifikasi terukur terpisah dari
kesalahan deteksi. Evaluasi menyeluruh (end-to-end) memakai detektor Pan-Tompkins sendiri.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime

import numpy as np
import sklearn
from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split

from src.classifier import MODEL_DIR, SEED, WeightedClassifier, make_model, predict, save_model, tune_decision_weights
from src.data_loader import CLASSES, DS1, DS2, load_record
from src.evaluation import evaluate, to_json
from src.features import FEATURE_GROUPS, FEATURE_NAMES, FEATURES, GROUP_LABELS, extract_features
from src.preprocessing import STANDARD, preprocess
from src.qrs_detection import match_beats, pan_tompkins
from src.synthetic import SYN_DS1, SYN_DS2, synthetic_patient


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_set(names, synthetic: bool):
    out = []
    for nm in names:
        rec = synthetic_patient(nm) if synthetic else load_record(nm)
        clean = preprocess(rec.signal, rec.fs, **STANDARD)
        out.append((rec, clean))
    return out


def build_dataset(items):
    X, y, g = [], [], []
    for rec, clean in items:
        F = extract_features(clean, rec.r_peaks, rec.fs)  # RR dihitung dari semua detak beranotasi
        keep = np.isin(rec.labels, CLASSES)               # F dan Q dikeluarkan dari klasifikasi
        X.append(F[keep])
        y.append(rec.labels[keep])
        g.append(np.full(int(keep.sum()), rec.uid))
    return np.vstack(X), np.concatenate(y), np.concatenate(g)


def counts(y) -> dict:
    return {c: int(np.sum(y == c)) for c in CLASSES}


def svm_subset(X, y, cap_n: int, seed: int):
    """SVM RBF berskala kuadratik: ambil semua S dan V, dan sebagian N (hanya data latih)."""
    rng = np.random.default_rng(seed)
    idx_n = np.flatnonzero(y == "N")
    if len(idx_n) > cap_n:
        idx_n = rng.choice(idx_n, cap_n, replace=False)
    idx = np.concatenate([idx_n, np.flatnonzero(y != "N")])
    return X[idx], y[idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="pakai pasien sintetis, bukan MIT-BIH")
    ap.add_argument("--tune", action="store_true", help="setel parameter RF dengan GroupKFold pada DS1")
    ap.add_argument("--svm-cap", type=int, default=8000, help="jumlah maksimum detak N untuk melatih SVM")
    ap.add_argument("--no-weights", action="store_true", help="tanpa penyetelan bobot keputusan per kelas")
    args = ap.parse_args()
    t_start = time.time()
    source = "sintetis" if args.synthetic else "MIT-BIH"

    log(f"Memuat {source}: DS1 (latih) dan DS2 (uji)…")
    ds1 = load_set(SYN_DS1 if args.synthetic else DS1, args.synthetic)
    ds2 = load_set(SYN_DS2 if args.synthetic else DS2, args.synthetic)
    Xtr, ytr, gtr = build_dataset(ds1)
    Xte, yte, _ = build_dataset(ds2)
    log(f"DS1: {len(ytr):,} detak {counts(ytr)} | DS2: {len(yte):,} detak {counts(yte)}")

    rf_params = {}
    tuning = None
    if args.tune:
        log("Menyetel Random Forest dengan validasi silang dikelompokkan per rekaman…")
        gs = GridSearchCV(make_model("rf", n_estimators=100),
                          {"max_depth": [None, 20, 10], "min_samples_leaf": [1, 2, 5]},
                          cv=GroupKFold(n_splits=5), scoring="f1_macro", n_jobs=-1)
        gs.fit(Xtr, ytr, groups=gtr)
        rf_params = gs.best_params_
        tuning = {"best_params": rf_params, "best_cv_f1_macro": float(gs.best_score_)}
        log(f"Parameter terbaik: {rf_params} (F1 makro CV {gs.best_score_:.3f})")

    meta = {"source": source, "fs": 360, "features": FEATURE_NAMES, "classes": CLASSES,
            "created": datetime.now().isoformat(timespec="seconds"),
            "sklearn": sklearn.__version__, "python": platform.python_version()}
    models, results = {}, {}
    subset = (lambda X, y: svm_subset(X, y, args.svm_cap, SEED))
    for name in ("rf", "svm", "knn"):
        t0 = time.time()
        params = rf_params if name == "rf" else {}
        base = make_model(name, **params)
        Xf, yf = subset(Xtr, ytr) if name == "svm" else (Xtr, ytr)
        base.fit(Xf, yf)
        ev_raw = evaluate(yte, predict(base, Xte))
        tw = None
        model = base
        if not args.no_weights:
            # Bobot keputusan disetel HANYA pada DS1 (GroupKFold per rekaman), lalu dipakai apa adanya di DS2.
            tw = tune_decision_weights(name, Xtr, ytr, gtr, fit_subset=subset if name == "svm" else None, params=params)
            model = WeightedClassifier(base, tw["weights"])
        ev = evaluate(yte, predict(model, Xte))
        results[name] = to_json(ev) | {"train_seconds": round(time.time() - t0, 1), "n_train": int(len(yf)),
                                       "macro_f1_tanpa_bobot": ev_raw["macro_f1"],
                                       "f1_S_tanpa_bobot": float(ev_raw["per_class"].query("kelas == 'S'")["f1"].iloc[0]),
                                       "bobot": tw}
        models[name] = model
        save_model(model, name, meta | {"macro_f1_ds2": ev["macro_f1"], "bobot": tw["weights"] if tw else None})
        log(f"{name}: F1 makro DS2 = {ev['macro_f1']:.3f} (tanpa bobot {ev_raw['macro_f1']:.3f}), "
            f"akurasi {ev['accuracy']:.3f}, bobot {tw['weights'] if tw else '-'} ({time.time() - t0:.0f} s)")

    log("Membandingkan kelompok fitur (RM 3)…")
    groups = {}
    for g, idx in FEATURE_GROUPS.items():
        m = make_model("rf", n_estimators=100).fit(Xtr[:, idx], ytr)
        groups[f"Hanya {GROUP_LABELS[g].lower()}"] = evaluate(yte, m.predict(Xte[:, idx]))["macro_f1"]
    groups[f"Gabungan ({len(FEATURE_NAMES)} fitur)"] = results["rf"]["macro_f1_tanpa_bobot"]
    base30 = list(range(30))  # 30 fitur awal, tanpa fitur relatif pasien
    m30 = make_model("rf", **rf_params).fit(Xtr[:, base30], ytr)
    ablation = {"30 fitur awal, tanpa bobot": evaluate(yte, m30.predict(Xte[:, base30]))["macro_f1"],
                f"{len(FEATURE_NAMES)} fitur, tanpa bobot": results["rf"]["macro_f1_tanpa_bobot"],
                f"{len(FEATURE_NAMES)} fitur + bobot keputusan": results["rf"]["macro_f1"]}

    log("Membandingkan dengan pembagian acak per detak…")
    Xall, yall = np.vstack([Xtr, Xte]), np.concatenate([ytr, yte])
    a_tr, a_te, b_tr, b_te = train_test_split(Xall, yall, test_size=0.5, stratify=yall, random_state=SEED)
    ev_rand = evaluate(b_te, make_model("rf", n_estimators=100).fit(a_tr, b_tr).predict(a_te))
    # Perbandingan adil: keduanya Random Forest tanpa bobot keputusan.
    split = {"acak_macro_f1": ev_rand["macro_f1"], "acak_f1_S": float(ev_rand["per_class"].query("kelas == 'S'")["f1"].iloc[0]),
             "pasien_macro_f1": results["rf"]["macro_f1_tanpa_bobot"], "pasien_f1_S": results["rf"]["f1_S_tanpa_bobot"]}

    log("Evaluasi menyeluruh dengan detektor Pan-Tompkins sendiri pada DS2…")
    per_rec, yt, yp = [], [], []
    TP = FP = FN = 0
    for rec, clean in ds2:
        pt = pan_tompkins(clean, rec.fs, refine_on=clean)
        pairs, tp, fp, fn = match_beats(pt.peaks, rec.r_peaks, rec.fs)
        TP, FP, FN = TP + tp, FP + fp, FN + fn
        per_rec.append({"rekaman": rec.name, "TP": tp, "FP": fp, "FN": fn,
                        "sensitivitas": tp / (tp + fn) if tp + fn else None,
                        "presisi": tp / (tp + fp) if tp + fp else None})
        F = extract_features(clean, pt.peaks, rec.fs)
        pred = predict(models["rf"], F)
        for i, j in pairs:
            if rec.labels[j] in CLASSES:
                yt.append(rec.labels[j])
                yp.append(pred[i])
    ev_e2e = evaluate(np.array(yt), np.array(yp))

    out = {
        "meta": meta, "counts": {"DS1": counts(ytr), "DS2": counts(yte)}, "tuning": tuning,
        "models": results, "feature_groups": groups, "split": split, "ablation_rf": ablation,
        "importance": [{"fitur": FEATURES[i][0], "kelompok": GROUP_LABELS[FEATURES[i][1]], "nilai": float(v)}
                       for i, v in enumerate(getattr(models["rf"], "model", models["rf"]).feature_importances_)],
        "end_to_end": {"sensitivitas": TP / (TP + FN), "presisi": TP / (TP + FP), "TP": TP, "FP": FP, "FN": FN,
                       "evaluasi": to_json(ev_e2e), "per_rekaman": per_rec},
    }
    MODEL_DIR.mkdir(exist_ok=True)
    (MODEL_DIR / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Selesai dalam {time.time() - t_start:.0f} s. Model dan hasil tersimpan di {MODEL_DIR}.")
    log(f"Deteksi R DS2: Se {out['end_to_end']['sensitivitas']:.4f}, PPV {out['end_to_end']['presisi']:.4f}; "
        f"F1 makro end-to-end {ev_e2e['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
