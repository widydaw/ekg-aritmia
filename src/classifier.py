"""Pelatihan, penyimpanan, dan prediksi model klasifikasi detak."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_NAMES = {"rf": "Random Forest", "svm": "SVM (RBF)", "knn": "k-NN"}
SEED = 42


def make_model(name: str, seed: int = SEED, **params):
    """Random Forest tidak butuh normalisasi; SVM dan k-NN wajib memakai StandardScaler."""
    if name == "rf":
        kw = dict(n_estimators=200, class_weight="balanced", min_samples_leaf=2, n_jobs=-1, random_state=seed)
        kw.update(params)
        return RandomForestClassifier(**kw)
    if name == "svm":
        kw = dict(kernel="rbf", C=4.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed)
        kw.update(params)
        return make_pipeline(StandardScaler(), SVC(**kw))
    if name == "knn":
        kw = dict(n_neighbors=5)
        kw.update(params)
        return make_pipeline(StandardScaler(), KNeighborsClassifier(**kw))
    raise ValueError(f"Model tidak dikenal: {name}")


class WeightedClassifier:
    """Model + bobot keputusan per kelas: kelas = argmax(probabilitas × bobot).

    Bobot disetel dengan validasi silang per rekaman pada DS1 saja (lihat tune_decision_weights),
    agar kelas minoritas S tidak tenggelam oleh N tanpa mengintip data uji.
    """

    def __init__(self, model, weights):
        self.model = model
        self.weights = np.asarray(weights, float)
        self.classes_ = model.classes_

    def predict_proba(self, X):
        p = self.model.predict_proba(X) * self.weights
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.model.predict_proba(X) * self.weights, axis=1)]


def tune_decision_weights(name, X, y, groups, fit_subset=None, params=None, n_splits=5,
                          grid_s=(1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24), grid_v=(0.75, 1, 1.5, 2, 3, 4)):
    """Cari bobot (N=1, S, V) yang memaksimalkan F1 makro pada prediksi out-of-fold GroupKFold di DS1."""
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupKFold

    X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
    oof, classes = np.zeros((len(y), 3)), None
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups):
        Xf, yf = (fit_subset(X[tr], y[tr]) if fit_subset else (X[tr], y[tr]))
        m = make_model(name, **(params or {})).fit(Xf, yf)
        classes = m.classes_
        oof[va] = m.predict_proba(X[va])
    best = (-1.0, (1.0, 1.0, 1.0))
    for ws in grid_s:
        for wv in grid_v:
            w = np.array([1.0, ws, wv])
            f1 = f1_score(y, classes[np.argmax(oof * w, axis=1)], average="macro")
            if f1 > best[0]:
                best = (f1, (1.0, float(ws), float(wv)))
    base = f1_score(y, classes[np.argmax(oof, axis=1)], average="macro")
    return {"weights": list(best[1]), "cv_f1_macro": float(best[0]), "cv_f1_macro_tanpa_bobot": float(base)}


def predict(model, features) -> np.ndarray:
    """Antarmuka yang disepakati: fitur → label kelas tiap detak."""
    X = np.asarray(features)
    return model.predict(X) if len(X) else np.array([], dtype=str)


def predict_proba(model, features):
    X = np.asarray(features)
    if not len(X) or not hasattr(model, "predict_proba"):
        return None
    return model.predict_proba(X)


def save_model(model, name: str, meta: dict | None = None, model_dir: Path = MODEL_DIR) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{name}.joblib"
    joblib.dump({"model": model, "meta": meta or {}}, path)
    return path


def load_model(name: str, model_dir: Path = MODEL_DIR):
    """Kembalikan (model, meta), atau (None, None) bila berkas belum ada."""
    path = model_dir / f"{name}.joblib"
    if not path.exists():
        return None, None
    obj = joblib.load(path)
    return obj["model"], obj.get("meta", {})
