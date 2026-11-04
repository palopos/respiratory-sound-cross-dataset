"""R1 Fase 2 (R3.5, R4.CS4, R2 resumen): calibracion de probabilidades frente a transferencia de umbral.

Por modelo, en validacion (origen) y test (destino):
  - Brier, ECE (15 bins de igual anchura) y ECE adaptativo (15 bins de igual masa)
  - calibracion global (media de p - prevalencia), intercepto y pendiente de calibracion (y ~ logit p)
  - Platt ajustado SOLO en validacion de origen y aplicado al destino (no usa etiquetas de test)
Transferencia de umbral (descriptiva):
  - umbral val_best_f1, percentil que ocupa en los scores del test, tasa de positivos predichos frente a prevalencia
  - umbral "oraculo" que maximiza F1 en test y su F1: SOLO descriptivo, no se usa para seleccionar nada
IC 95 % por paciente (B = 1000) para Brier y ECE en test.

Uso: python codigo/fase2/r1_fase2_calibration.py   (desde la raiz del repositorio)
Salidas en analisis/r1_20261101_fase2_v2/: calibracion.csv, fiabilidad_bins.csv, figuras/fiabilidad_*.{png,pdf}
"""
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1_fase2_common import (PROTOCOL_LABEL, attach_metadata, load_ensemble, model_registry,  # noqa: E402
                             target_of, val_best_f1_threshold, point_metrics)

OUT = Path(os.environ.get("R1_FASE2_OUT", "analisis/r1_20261101_fase2_v2"))  # R1_FASE2_OUT: version nueva sin sobrescribir
N_BINS, B, SEED = 15, 1000, 20261029
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def ece(y, p, adaptive=False):
    if adaptive:
        edges = np.unique(np.quantile(p, np.linspace(0, 1, N_BINS + 1)))
    else:
        edges = np.linspace(0, 1, N_BINS + 1)
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    total = 0.0
    for b in range(len(edges) - 1):
        mask = bins == b
        if mask.any():
            total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return total


def calibration_slope_intercept(y, p):
    if y.min() == y.max():
        return np.nan, np.nan
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def calib_block(y, p, prefix):
    slope, intercept = calibration_slope_intercept(y, p)
    return {f"{prefix}_brier": brier_score_loss(y, p), f"{prefix}_ece": ece(y, p),
            f"{prefix}_ece_adaptive": ece(y, p, adaptive=True), f"{prefix}_mean_prob": float(p.mean()),
            f"{prefix}_prevalence": float(y.mean()), f"{prefix}_calib_in_the_large": float(p.mean() - y.mean()),
            f"{prefix}_calib_slope": slope, f"{prefix}_calib_intercept": intercept}


def reliability_bins(y, p):
    edges = np.linspace(0, 1, N_BINS + 1)
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, N_BINS - 1)
    rows = []
    for b in range(N_BINS):
        mask = bins == b
        if mask.any():
            rows.append({"bin": b, "bin_lo": edges[b], "bin_hi": edges[b + 1], "n": int(mask.sum()),
                         "mean_prob": float(p[mask].mean()), "observed_rate": float(y[mask].mean())})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figuras").mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows, bin_rows, curves = [], [], {}
    for r in model_registry().itertuples():
        val, test, thr = load_ensemble(r.run_dir)
        test = attach_metadata(test, target_of(r.protocol), "test")
        yv, pv = val["y_true"].to_numpy(), val["prob"].to_numpy()
        yt, pt = test["y_true"].to_numpy(), test["prob"].to_numpy()

        platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(pv).reshape(-1, 1), yv)
        pt_platt = platt.predict_proba(logit(pt).reshape(-1, 1))[:, 1]
        oracle_thr = val_best_f1_threshold(yt, pt)
        at_val = point_metrics(yt, pt, thr)
        at_oracle = point_metrics(yt, pt, oracle_thr)

        row = {"protocol": r.protocol, "model": r.model, "evidence": r.evidence, "group": r.group,
               **calib_block(yv, pv, "val"), **calib_block(yt, pt, "test"),
               "test_platt_brier": brier_score_loss(yt, pt_platt), "test_platt_ece": ece(yt, pt_platt),
               "test_platt_calib_in_the_large": float(pt_platt.mean() - yt.mean()),
               "val_threshold": thr, "val_threshold_percentile_in_test_scores": percentileofscore(pt, thr, kind="strict"),
               "val_threshold_percentile_in_val_scores": percentileofscore(pv, thr, kind="strict"),
               "test_pred_positive_rate_at_val_thr": at_val["pred_positive_rate"],
               "val_pred_positive_rate_at_val_thr": float((pv >= thr).mean()),
               "test_f1_at_val_thr": at_val["f1"], "test_sens_at_val_thr": at_val["sensitivity"],
               "test_spec_at_val_thr": at_val["specificity"],
               "oracle_test_threshold": oracle_thr, "oracle_test_f1": at_oracle["f1"],
               "test_f1_all_positive": 2 * yt.mean() / (1 + yt.mean()),
               "test_auroc": at_val["auroc"]}

        by_patient = test.groupby("patient_id").indices
        groups = list(by_patient.values())
        reps = []
        for _ in range(B):
            idx = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
            reps.append((brier_score_loss(yt[idx], pt[idx]), ece(yt[idx], pt[idx])))
        reps = np.array(reps)
        row["test_brier_patient_lo"], row["test_brier_patient_hi"] = np.percentile(reps[:, 0], [2.5, 97.5])
        row["test_ece_patient_lo"], row["test_ece_patient_hi"] = np.percentile(reps[:, 1], [2.5, 97.5])
        rows.append(row)
        for split, (y, p) in (("validation", (yv, pv)), ("test", (yt, pt))):
            for b in reliability_bins(y, p):
                bin_rows.append({"protocol": r.protocol, "model": r.model, "split": split, **b})
        curves[(r.protocol, r.model)] = (yv, pv, yt, pt, thr)
        print(f"{r.protocol:21s} {r.model[:44]:44s} ECE val {row['val_ece']:.3f} test {row['test_ece']:.3f} | "
              f"thr {thr:.3f} pct-test {row['val_threshold_percentile_in_test_scores']:.1f} | "
              f"F1 val-thr {row['test_f1_at_val_thr']:.3f} oracle {row['oracle_test_f1']:.3f}", flush=True)

    pd.DataFrame(rows).to_csv(OUT / "calibracion.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(OUT / "fiabilidad_bins.csv", index=False)

    # Curvas de fiabilidad + histograma de scores para los modelos principales.
    models = ["CNN scratch", "YAMNet trainable head (loop pad)", "wav2vec 2.0 fine-tuned (R1 fixed)"]  # familias principales R1
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), gridspec_kw={"height_ratios": [3, 1.3]})
    for j, protocol in enumerate(PROTOCOL_LABEL):
        ax, axh = axes[0, j], axes[1, j]
        ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--")
        for model, color in zip(models, colors):
            if (protocol, model) not in curves:
                continue
            _, _, yt, pt, thr = curves[(protocol, model)]
            b = pd.DataFrame(reliability_bins(yt, pt))
            b = b[b["n"] >= 10]
            ax.plot(b["mean_prob"], b["observed_rate"], "o-", color=color, ms=4, lw=1.6, label=model)
            axh.hist(pt, bins=30, range=(0, 1), histtype="step", color=color, lw=1.4)
            axh.axvline(thr, color=color, ls=":", lw=1.4)
        ax.axhline(curves[(protocol, models[0])][2].mean(), color="0.3", lw=0.8, ls=":")
        ax.set_title(PROTOCOL_LABEL[protocol].replace("->", "→"), fontsize=13)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("Mean predicted probability", fontsize=11)
        if j == 0:
            ax.set_ylabel("Observed adventitious rate", fontsize=11)
            axh.set_ylabel("Test events", fontsize=11)
        axh.set_xlabel("Predicted probability (dotted: validation threshold)", fontsize=10)
        ax.tick_params(labelsize=10); axh.tick_params(labelsize=9)
    axes[0, 0].legend(fontsize=9, loc="upper left", frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figuras" / f"fiabilidad_modelos_principales.{ext}", dpi=300)
    print(f">>> {len(rows)} modelos -> {OUT}")


if __name__ == "__main__":
    main()
