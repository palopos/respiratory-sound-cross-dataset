"""R1 Fase 2 (R2.4, R3.4, R4.CS3, R4.CS5): analisis de factores del desplazamiento entre datasets.

Sin reentrenar, desde predicciones guardadas y con el umbral de validacion de cada modelo:
  1. Prevalencia: AUPRC estandarizado a prevalencia 0.5 (reponderando clases) y lift = AUPRC / prevalencia.
  2. Duracion: metricas por franja de duracion y AUROC/AUPRC "emparejado por duracion", que repondera los
     eventos de test para que la distribucion de duracion de CADA clase iguale la del entrenamiento de origen.
  3. Subtipos: sensibilidad por etiqueta original (Crackle, Wheeze, ...) y AUROC subtipo frente a Normal.
  4. Metadatos: AUROC por dispositivo / localizacion (ICBHI) y por localizacion / edad (SPRSound).
IC 95 % por paciente (B = 1000) para las metricas reponderadas.

Uso: python codigo/fase2/r1_fase2_subgroups.py   (desde la raiz del repositorio)
Salidas en analisis/r1_20261101_fase2_v2/: factores_prevalencia_duracion.csv, por_franja_duracion.csv,
  por_subtipo.csv, por_estrato_metadatos.csv
"""
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1_fase2_common import (attach_metadata, event_metadata, load_ensemble, model_registry,  # noqa: E402
                             source_of, target_of)

OUT = Path(os.environ.get("R1_FASE2_OUT", "analisis/r1_20261101_fase2_v2"))  # R1_FASE2_OUT: version nueva sin sobrescribir
DUR_EDGES = [0.0, 0.975, 2.0, 4.0, np.inf]
DUR_LABELS = ["<0.975 s", "0.975-2 s", "2-4 s", ">=4 s"]
B, SEED = 1000, 20261029
MIN_PER_CLASS = 10


def dur_bin(d):
    return pd.cut(d, DUR_EDGES, labels=DUR_LABELS, right=False)


def prevalence_weights(y, target=0.5):
    pi = y.mean()
    return np.where(y == 1, target / pi, (1 - target) / (1 - pi))


def duration_match_weights(test, source_train):
    """Peso por (clase, franja) = cuota en entrenamiento de origen / cuota en test, dentro de cada clase."""
    w = np.zeros(len(test))
    for cls in (0, 1):
        src = dur_bin(source_train.loc[source_train["y"] == cls, "duration_s"]).value_counts(normalize=True)
        mask = (test["y_true"] == cls).to_numpy()
        tgt_bins = dur_bin(test.loc[mask, "duration_s"])
        tgt = tgt_bins.value_counts(normalize=True)
        ratio = (src / tgt).replace([np.inf], np.nan).fillna(0.0)
        w[mask] = tgt_bins.map(ratio).astype(float).to_numpy()
    return w


def weighted_metrics(y, p, w):
    if y.min() == y.max() or w[y == 1].sum() == 0 or w[y == 0].sum() == 0:
        return np.nan, np.nan
    return roc_auc_score(y, p, sample_weight=w), average_precision_score(y, p, sample_weight=w)


def safe_auroc(y, p):
    y = np.asarray(y)
    if (y == 1).sum() < MIN_PER_CLASS or (y == 0).sum() < MIN_PER_CLASS:
        return np.nan
    return roc_auc_score(y, p)


def main():
    rng = np.random.default_rng(SEED)
    source_train = {}
    for ds in ("icbhi", "sprsound"):
        m = event_metadata(ds, "train")
        m["y"] = (m["label_name"] != "Normal").astype(int)
        source_train[ds] = m

    fac_rows, bin_rows, sub_rows, strat_rows = [], [], [], []
    for r in model_registry().itertuples():
        if r.group in ("r1_padding_min0975",):
            continue  # su test ya esta restringido por duracion
        _, test, thr = load_ensemble(r.run_dir)
        target = target_of(r.protocol)
        test = attach_metadata(test, target, "test")
        y, p = test["y_true"].to_numpy(), test["prob"].to_numpy()
        pred = p >= thr
        w_prev = prevalence_weights(y)
        w_dur = duration_match_weights(test, source_train[source_of(r.protocol)])
        w_both = w_dur * prevalence_weights(y) / np.where(y == 1, w_dur[y == 1].mean(), w_dur[y == 0].mean())

        point = {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
                 "auprc_prev05": average_precision_score(y, p, sample_weight=w_prev)}
        point["auroc_dur_matched"], point["auprc_dur_matched"] = weighted_metrics(y, p, w_dur)
        _, point["auprc_dur_matched_prev05"] = weighted_metrics(y, p, w_both)

        groups = list(test.groupby("patient_id").indices.values())
        reps = []
        for _ in range(B):
            idx = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
            yi, pi = y[idx], p[idx]
            if yi.min() == yi.max():
                reps.append([np.nan] * 4)
                continue
            a_d, p_d = weighted_metrics(yi, pi, w_dur[idx])
            reps.append([average_precision_score(yi, pi, sample_weight=prevalence_weights(yi)), a_d, p_d,
                         average_precision_score(yi, pi) / yi.mean()])
        reps = np.array(reps)
        lo, hi = np.nanpercentile(reps, 2.5, axis=0), np.nanpercentile(reps, 97.5, axis=0)
        fac_rows.append({"protocol": r.protocol, "model": r.model, "group": r.group, "n_events": len(y),
                         "prevalence": y.mean(), **point, "auprc_lift": point["auprc"] / y.mean(),
                         "auprc_prev05_lo": lo[0], "auprc_prev05_hi": hi[0],
                         "auroc_dur_matched_lo": lo[1], "auroc_dur_matched_hi": hi[1],
                         "auprc_dur_matched_lo": lo[2], "auprc_dur_matched_hi": hi[2],
                         "auprc_lift_lo": lo[3], "auprc_lift_hi": hi[3],
                         "dur_weight_max": float(w_dur.max())})

        bins = dur_bin(test["duration_s"])
        for b in DUR_LABELS:
            mask = (bins == b).to_numpy()
            if not mask.any():
                continue
            yb, pb, predb = y[mask], p[mask], pred[mask]
            bin_rows.append({"protocol": r.protocol, "model": r.model, "group": r.group, "duration_bin": b,
                             "n_events": int(mask.sum()), "n_patients": test.loc[mask, "patient_id"].nunique(),
                             "prevalence": yb.mean(), "auroc": safe_auroc(yb, pb), "mean_prob": pb.mean(),
                             "sensitivity": predb[yb == 1].mean() if (yb == 1).any() else np.nan,
                             "specificity": 1 - predb[yb == 0].mean() if (yb == 0).any() else np.nan})

        normal = (test["label_name"] == "Normal").to_numpy()
        for lab in sorted(test["label_name"].unique()):
            mask = (test["label_name"] == lab).to_numpy()
            row = {"protocol": r.protocol, "model": r.model, "group": r.group, "label_name": lab,
                   "n_events": int(mask.sum()), "pred_positive_rate": pred[mask].mean(), "mean_prob": p[mask].mean()}
            if lab != "Normal":
                both = mask | normal
                row["auroc_vs_normal"] = safe_auroc(y[both], p[both])
            sub_rows.append(row)

        strata = ["equipment", "chest_location"] if target == "icbhi" else ["location", "age_band"]
        if target == "sprsound":
            test["age_band"] = pd.cut(test["age"], [0, 3, 6, 12, 100], labels=["<3", "3-6", "6-12", ">=12"], right=False)
        for col in strata:
            for val, sub in test.groupby(col, observed=True):
                idx = sub.index.to_numpy()
                strat_rows.append({"protocol": r.protocol, "model": r.model, "group": r.group, "stratum": col,
                                   "value": str(val), "n_events": len(idx), "n_patients": sub["patient_id"].nunique(),
                                   "prevalence": y[idx].mean(), "auroc": safe_auroc(y[idx], p[idx]),
                                   "sensitivity": pred[idx][y[idx] == 1].mean() if (y[idx] == 1).any() else np.nan,
                                   "specificity": 1 - pred[idx][y[idx] == 0].mean() if (y[idx] == 0).any() else np.nan})
        print(f"{r.protocol:21s} {r.model[:44]:44s} AUROC {point['auroc']:.4f} dur-matched {point['auroc_dur_matched']:.4f} "
              f"| AUPRC {point['auprc']:.4f} prev0.5 {point['auprc_prev05']:.4f}", flush=True)

    pd.DataFrame(fac_rows).to_csv(OUT / "factores_prevalencia_duracion.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(OUT / "por_franja_duracion.csv", index=False)
    pd.DataFrame(sub_rows).to_csv(OUT / "por_subtipo.csv", index=False)
    pd.DataFrame(strat_rows).to_csv(OUT / "por_estrato_metadatos.csv", index=False)
    print(f">>> {len(fac_rows)} modelos -> {OUT}")


if __name__ == "__main__":
    main()
