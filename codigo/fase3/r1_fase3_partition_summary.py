"""R1 Fase 3 (R2.3, R3.2): sensibilidad de los resultados a la particion por paciente.

Por familia (YAMNet head con relleno loop; CNN desde cero), protocolo y semilla de particion (42, 101, 202, 303):
ensemble de 3 semillas, umbral val_best_f1 de validacion, AUROC/AUPRC/F1 en test e IC 95 % por paciente (B = 1000).
Tambien la diferencia YAMNet - CNN y la asimetria (AUROC ICBHI->SPRSound menos SPRSound->ICBHI) por particion.
La particion 42 usa las ejecuciones existentes (YAMNet loop de la Fase 1; CNN originales del paper).

Uso: python codigo/fase3/r1_fase3_partition_summary.py   (desde la raiz del repositorio)
Salidas: analisis/r1_20261029_fase3/particiones_metricas.csv, particiones_resumen_familia.csv
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fase2"))
from r1_fase2_common import ORIG, PROTOCOL_LABEL, load_ensemble, point_metrics, recording_of  # noqa: E402

OUT = Path("analisis/r1_20261029_fase3")
SEEDS = [42, 101, 202, 303]
B, SEED = 1000, 20261029
YAMNET_SETUP = ("yamnet/yamnet_google-yamnet-tfhub_frozen-backbone-trainable-head_trainable-mlp_"
                "classification-head-only")
CNN_SETUP = "cnn/cnn_biocas-resnext-attention_scratch_trainable-mlp"
CNN_ORIGINAL = {
    "sprsound_to_sprsound": "sprsound-to-sprsound/{s}/split_seed_42/run_20260521_091345",
    "icbhi_to_icbhi": "icbhi-to-icbhi/{s}/split_seed_42/run_20260521_090606",
    "icbhi_to_sprsound": "icbhi-to-sprsound/{s}/split_seed_42/run_20260521_132020",
    "sprsound_to_icbhi": "sprsound-to-icbhi/{s}/split_seed_42/run_20260717_105233_corrected-full-icbhi-test",
}


def run_dir(family, protocol, seed):
    slug = protocol.replace("_", "-")
    if family == "YAMNet head (loop)":
        if seed == 42:
            return Path(f"modelos/r1-20261029-yamnet-head-padding-loop/{slug}/{YAMNET_SETUP}/split_seed_42/r1_20261029_loop")
        return Path(f"modelos/r1-20261029-yamnet-head-partitions/{slug}/{YAMNET_SETUP}/split_seed_{seed}/r1_20261029_loop_split{seed}")
    if seed == 42:
        return ORIG / "modelos/00-cnn-scratch-baseline" / CNN_ORIGINAL[protocol].format(s=CNN_SETUP)
    src, tgt = protocol.split("_to_")
    suffix = "" if src == tgt else "_crosseval"
    return Path(f"modelos/r1-20261029-cnn-partitions/{slug}/{CNN_SETUP}/split_seed_{seed}/r1_20261029_split{seed}{suffix}")


def patients_for(test, protocol, seed):
    target = protocol.split("_to_")[1]
    name = {"icbhi": "0 ICBHI", "sprsound": "0 SPRSound"}[target]
    split = pd.read_csv(f"ficheros/splits/{name}/splits_patient_independent_seed{seed}.csv")
    mapping = dict(zip(split["recording_id"], split["patient_group"]))
    pid = test["stem"].map(recording_of).map(mapping)
    if pid.isna().any():
        raise ValueError(f"Eventos sin paciente en {protocol} particion {seed}")
    test_recordings = set(split.loc[split["split"] == "test", "recording_id"])
    if not test["stem"].map(recording_of).isin(test_recordings).all():
        raise ValueError(f"Eventos de test fuera del split de test en {protocol} particion {seed}")
    return pid.to_numpy()


def main():
    rng = np.random.default_rng(SEED)
    rows, reps_store = [], {}
    for family in ("YAMNet head (loop)", "CNN scratch"):
        for protocol in PROTOCOL_LABEL:
            for seed in SEEDS:
                d = run_dir(family, protocol, seed)
                complete = [s for s in d.glob("seed_*") if (s / "predictions" / "predicciones_test.csv").exists()
                            and (s / "predictions" / "predicciones_validation.csv").exists()]
                if len(complete) < 3:
                    continue  # ejecucion inexistente o todavia en curso
                _, test, thr = load_ensemble(d)
                pid = patients_for(test, protocol, seed)
                y, p = test["y_true"].to_numpy(), test["prob"].to_numpy()
                m = point_metrics(y, p, thr)
                groups = list(pd.Series(range(len(y))).groupby(pid).indices.values())
                reps = []
                for _ in range(B):
                    idx = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
                    reps.append(roc_auc_score(y[idx], p[idx]) if y[idx].min() != y[idx].max() else np.nan)
                reps = np.array(reps)
                reps_store[(family, protocol, seed)] = reps
                rows.append({"family": family, "protocol": protocol, "partition_seed": seed,
                             "n_seeds": int(test["n_seeds"].iloc[0]), "n_events": len(y),
                             "n_patients": len(groups), "prevalence": y.mean(), "val_threshold": thr,
                             "auroc": m["auroc"], "auroc_lo": np.nanpercentile(reps, 2.5),
                             "auroc_hi": np.nanpercentile(reps, 97.5), "auprc": m["auprc"],
                             "auprc_lift": m["auprc"] / y.mean(), "f1": m["f1"],
                             "sensitivity": m["sensitivity"], "specificity": m["specificity"], "run_dir": str(d)})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "particiones_metricas.csv", index=False)

    summary = (df.groupby(["family", "protocol"])
               .agg(n_partitions=("auroc", "size"), auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
                    auroc_min=("auroc", "min"), auroc_max=("auroc", "max"),
                    auprc_lift_mean=("auprc_lift", "mean"), f1_mean=("f1", "mean"), f1_min=("f1", "min"),
                    f1_max=("f1", "max"))
               .reset_index())
    summary.to_csv(OUT / "particiones_resumen_familia.csv", index=False)

    # Asimetria por particion: AUROC(ICBHI->SPRSound) - AUROC(SPRSound->ICBHI) e IC de la diferencia
    # (remuestras independientes porque los destinos son distintos).
    asym = []
    for family in df["family"].unique():
        for seed in SEEDS:
            a, b = (family, "icbhi_to_sprsound", seed), (family, "sprsound_to_icbhi", seed)
            if a in reps_store and b in reps_store:
                diff = reps_store[a] - reps_store[b]
                pa = df.query("family == @family and protocol == 'icbhi_to_sprsound' and partition_seed == @seed")["auroc"].iloc[0]
                pb = df.query("family == @family and protocol == 'sprsound_to_icbhi' and partition_seed == @seed")["auroc"].iloc[0]
                asym.append({"family": family, "partition_seed": seed, "auroc_icbhi_to_sprsound": pa,
                             "auroc_sprsound_to_icbhi": pb, "asymmetry": pa - pb,
                             "asymmetry_lo": np.nanpercentile(diff, 2.5), "asymmetry_hi": np.nanpercentile(diff, 97.5)})
    pd.DataFrame(asym).to_csv(OUT / "particiones_asimetria.csv", index=False)
    with pd.option_context("display.width", 220):
        print(df.drop(columns="run_dir").round(4).to_string(index=False))
        print(summary.round(4).to_string(index=False))
        print(pd.DataFrame(asym).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
