"""R1 Fase 2 (R2.2, R3.2, R4.CS1, R4.CS3, R4.CS4): bootstrap por paciente.

Para cada modelo: ensemble de semillas, umbral val_best_f1 fijado en validacion y, sobre test,
remuestreo de pacientes con reemplazo (B = 2000). Todos los modelos con el mismo dataset destino
usan las mismas remuestras, de modo que las diferencias entre modelos son pareadas.
Tambien se recalcula el bootstrap por evento (B = 1000, semilla 12345) solo como referencia.

Uso: python codigo/fase2/r1_fase2_patient_bootstrap.py   (desde la raiz del repositorio)
Salidas en analisis/r1_20261101_fase2_v2/:
  bootstrap_por_paciente.csv      estimacion puntual + IC 95 % percentil (paciente y evento)
  comparaciones_pareadas.csv      diferencias pareadas (degradacion cross - intra, modelos, rellenos)
  bootstrap_replicas.npz          replicas por modelo (para figuras y analisis posteriores)
"""
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1_fase2_common import attach_metadata, load_ensemble, model_registry, point_metrics, target_of  # noqa: E402

B_PATIENT, SEED_PATIENT = 2000, 20261029
B_EVENT, SEED_EVENT = 1000, 12345
OUT = Path(os.environ.get("R1_FASE2_OUT", "analisis/r1_20261101_fase2_v2"))  # R1_FASE2_OUT: version nueva sin sobrescribir
METRICS = ["auroc", "auprc", "f1", "sensitivity", "specificity"]


def fast_metrics(y, p, thr):
    if y.min() == y.max():
        return [np.nan] * len(METRICS)
    pred = p >= thr
    tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
    fn = np.sum(~pred & (y == 1)); tn = np.sum(~pred & (y == 0))
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return [roc_auc_score(y, p), average_precision_score(y, p), f1, tp / (tp + fn), tn / (tn + fp)]


def ci(values):
    v = values[~np.isnan(values)]
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (np.nan, np.nan)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    registry = model_registry()
    loaded = {}
    for r in registry.itertuples():
        _, test, thr = load_ensemble(r.run_dir)
        loaded[r.model_id] = (attach_metadata(test, target_of(r.protocol), "test"), thr)

    # Remuestras de pacientes compartidas por dataset destino.
    patient_draws = {}
    rng = np.random.default_rng(SEED_PATIENT)
    for target in ("icbhi", "sprsound"):
        patients = np.array(sorted({pid for mid, (t, _) in loaded.items() if target_of(mid.split("|")[0]) == target
                                    for pid in t["patient_id"].unique()}, key=str))
        patient_draws[target] = (patients, rng.integers(0, len(patients), size=(B_PATIENT, len(patients))))

    rows, replicas = [], {}
    for r in registry.itertuples():
        test, thr = loaded[r.model_id]
        y, p = test["y_true"].to_numpy(), test["prob"].to_numpy()
        point = point_metrics(y, p, thr)
        patients, draws = patient_draws[target_of(r.protocol)]
        by_patient = test.groupby("patient_id").indices
        idx_of = [by_patient.get(pid, np.array([], dtype=int)) for pid in patients]
        reps = np.full((B_PATIENT, len(METRICS)), np.nan)
        for b in range(B_PATIENT):
            idx = np.concatenate([idx_of[k] for k in draws[b]])
            reps[b] = fast_metrics(y[idx], p[idx], thr)
        replicas[r.model_id] = reps
        ev_rng = np.random.default_rng(SEED_EVENT)
        ev_reps = np.array([fast_metrics(y[i], p[i], thr) for i in
                            (ev_rng.integers(0, len(y), len(y)) for _ in range(B_EVENT))])
        row = {"protocol": r.protocol, "model": r.model, "evidence": r.evidence, "group": r.group,
               "n_seeds": int(test["n_seeds"].iloc[0]), "n_events": len(y), "n_patients": test["patient_id"].nunique(),
               "prevalence": float(y.mean()), "val_threshold": thr, **point}
        for k, m in enumerate(METRICS):
            row[f"{m}_patient_lo"], row[f"{m}_patient_hi"] = ci(reps[:, k])
            row[f"{m}_event_lo"], row[f"{m}_event_hi"] = ci(ev_reps[:, k])
        row["auroc_patient_p_below_0_5"] = float(np.nanmean(reps[:, 0] < 0.5))
        row["n_invalid_patient_reps"] = int(np.isnan(reps[:, 0]).sum())
        rows.append(row)
        print(f"{r.protocol:21s} {r.model[:44]:44s} AUROC {point['auroc']:.4f} "
              f"[{row['auroc_patient_lo']:.4f}, {row['auroc_patient_hi']:.4f}] paciente | "
              f"[{row['auroc_event_lo']:.4f}, {row['auroc_event_hi']:.4f}] evento", flush=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "bootstrap_por_paciente.csv", index=False)
    np.savez_compressed(OUT / "bootstrap_replicas.npz", metrics=np.array(METRICS),
                        **{k.replace("|", "__").replace(" ", "_"): v for k, v in replicas.items()})

    # Comparaciones pareadas (mismo dataset destino => mismas remuestras de pacientes).
    def mid(protocol, model):
        return f"{protocol}|{model}"

    comparisons = []
    families = ["CNN scratch", "YAMNet frozen + LR", "YAMNet trainable head", "wav2vec 2.0 frozen + LR",
                "wav2vec 2.0 fine-tuned", "YAMNet trainable head (loop pad)", "wav2vec 2.0 fine-tuned (R1 fixed)"]
    for fam in families:  # R4.CS3: degradacion cross - intra sobre el mismo destino
        for target, cross, intra in (("sprsound", "icbhi_to_sprsound", "sprsound_to_sprsound"),
                                     ("icbhi", "sprsound_to_icbhi", "icbhi_to_icbhi")):
            comparisons.append(("degradacion_cross_menos_intra", fam, mid(cross, fam), mid(intra, fam)))
    for protocol in ("sprsound_to_sprsound", "icbhi_to_icbhi", "icbhi_to_sprsound", "sprsound_to_icbhi"):
        comparisons.append(("modelo", "YAMNet head - CNN", mid(protocol, "YAMNet trainable head"), mid(protocol, "CNN scratch")))
        comparisons.append(("modelo", "YAMNet head (loop) - CNN", mid(protocol, "YAMNet trainable head (loop pad)"),
                            mid(protocol, "CNN scratch")))
        comparisons.append(("modelo", "w2v2 R1 - CNN", mid(protocol, "wav2vec 2.0 fine-tuned (R1 fixed)"), mid(protocol, "CNN scratch")))
        comparisons.append(("modelo", "YAMNet head (loop) - w2v2 R1", mid(protocol, "YAMNet trainable head (loop pad)"),
                            mid(protocol, "wav2vec 2.0 fine-tuned (R1 fixed)")))
        for variant in ("loop pad", "noise pad"):
            base = "YAMNet trainable head (zero pad, R1 3 seeds)" if protocol.startswith("sprsound") else "YAMNet trainable head"
            comparisons.append(("relleno", f"{variant} - zero pad", mid(protocol, f"YAMNet trainable head ({variant})"), mid(protocol, base)))
    comparisons.append(("modelo", "w2v2 FT - CNN", mid("sprsound_to_icbhi", "wav2vec 2.0 fine-tuned"), mid("sprsound_to_icbhi", "CNN scratch")))
    comparisons.append(("modelo", "Mixup - baseline", mid("icbhi_to_sprsound", "YAMNet trainable head + Mixup"),
                        mid("icbhi_to_sprsound", "YAMNet trainable head")))
    comparisons.append(("modelo", "CORAL - baseline", mid("icbhi_to_sprsound", "YAMNet trainable head + CORAL"),
                        mid("icbhi_to_sprsound", "YAMNet trainable head")))
    point_by_id = summary.assign(model_id=summary["protocol"] + "|" + summary["model"]).set_index("model_id")
    comp_rows = []
    for kind, label, a, b in comparisons:
        if a not in replicas or b not in replicas:
            continue
        for k, m in enumerate(METRICS[:3]):
            diff = replicas[a][:, k] - replicas[b][:, k]
            lo, hi = ci(diff)
            comp_rows.append({"tipo": kind, "comparacion": label, "modelo_a": a, "modelo_b": b, "metric": m,
                              "diff_point": point_by_id.loc[a, m] - point_by_id.loc[b, m], "diff_lo": lo, "diff_hi": hi,
                              "p_diff_le_0": float(np.nanmean(diff <= 0))})
    pd.DataFrame(comp_rows).to_csv(OUT / "comparaciones_pareadas.csv", index=False)
    print(f">>> {len(summary)} modelos, {len(comp_rows)} comparaciones -> {OUT}")


if __name__ == "__main__":
    main()
