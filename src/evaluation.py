"""Confusion matrix dan metrik per kelas (satu lawan semua)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .data_loader import CLASSES


def evaluate(y_true, y_pred, labels=CLASSES) -> dict:
    """Antarmuka yang disepakati: kembalikan confusion matrix dan metrik per kelas.

    Sensitivitas = TP/(TP+FN), Presisi = TP/(TP+FP), Spesifisitas = TN/(TN+FP),
    F1 = 2·P·R/(P+R), F1 makro = rata-rata F1 semua kelas.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    rows = []
    for k, lab in enumerate(labels):
        tp = cm[k, k]
        fn = cm[k].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = total - tp - fn - fp
        se = tp / (tp + fn) if tp + fn else np.nan
        ppv = tp / (tp + fp) if tp + fp else np.nan
        sp = tn / (tn + fp) if tn + fp else np.nan
        f1 = 2 * se * ppv / (se + ppv) if (se + ppv) and np.isfinite(se + ppv) else 0.0
        rows.append({"kelas": lab, "jumlah": int(tp + fn), "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
                     "sensitivitas": se, "presisi": ppv, "spesifisitas": sp, "f1": f1})
    per = pd.DataFrame(rows)
    return {
        "cm": cm,
        "per_class": per,
        "macro_f1": float(per["f1"].mean()),
        "accuracy": float(np.trace(cm) / total) if total else float("nan"),
        "n": int(total),
    }


def to_json(ev: dict) -> dict:
    """Ubah hasil evaluate() menjadi bentuk yang dapat disimpan sebagai JSON."""
    return {
        "cm": ev["cm"].tolist(),
        "per_class": ev["per_class"].replace({np.nan: None}).to_dict(orient="records"),
        "macro_f1": ev["macro_f1"],
        "accuracy": ev["accuracy"],
        "n": ev["n"],
    }
