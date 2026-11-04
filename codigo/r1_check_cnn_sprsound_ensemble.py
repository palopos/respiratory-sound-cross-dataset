"""Reproduce el AUROC/AUPRC/F1 del ensemble CNN SPRSound -> SPRSound (Tabla 3 del paper).

Ensemble = media de prob_adventitious de las semillas 42-44; umbral = max F1 en validacion.
Uso: python codigo/r1_check_cnn_sprsound_ensemble.py  (desde la raiz del repositorio)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score

RUN = Path("ficheros/predicciones_originales/modelos/00-cnn-scratch-baseline/sprsound-to-sprsound/cnn/"
           "cnn_biocas-resnext-attention_scratch_trainable-mlp/split_seed_42/run_20260521_091345")
SEEDS = [42, 43, 44]


def load(split):
    frames = [pd.read_csv(RUN / f"seed_{s}/predictions/predicciones_{split}.csv").set_index("sample_path") for s in SEEDS]
    base = frames[0]
    for f in frames[1:]:
        assert f.index.equals(base.index) and (f["y_true"] == base["y_true"]).all()
    probs = np.column_stack([f["prob_adventitious"].to_numpy() for f in frames])
    return base["y_true"].to_numpy(), probs


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return thr[np.nanargmax(f1)]


yv, pv = load("validation")
yt, pt = load("test")
print(f"validation n={len(yv)}  test n={len(yt)}  test prevalence={yt.mean():.4f}")
for i, s in enumerate(SEEDS):
    print(f"seed {s}: test AUROC={roc_auc_score(yt, pt[:, i]):.4f}")
print(f"media AUROC por semilla = {np.mean([roc_auc_score(yt, pt[:, i]) for i in range(3)]):.4f}")
ev, et = pv.mean(1), pt.mean(1)
thr = best_f1_threshold(yv, ev)
print(f"ENSEMBLE: AUROC={roc_auc_score(yt, et):.4f}  AUPRC={average_precision_score(yt, et):.4f}  "
      f"thr_val={thr:.4f}  F1={f1_score(yt, (et >= thr).astype(int)):.4f}")
