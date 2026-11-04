"""R1 Fase 1: metricas de ensemble (media de probabilidades de las semillas) para las
ejecuciones YAMNet trainable head con distintos rellenos y para las ejecuciones originales.

Umbral = max F1 sobre las probabilidades medias de validacion (val_best_f1), aplicado a test.
Uso: python codigo/r1_padding_ensemble_metrics.py  (desde la raiz del repositorio)
Salida: analisis/r1_20261029_padding/padding_ensemble_metrics.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

OUT_DIR = Path("analisis/r1_20261029_padding")
HEAD = "yamnet/yamnet_google-yamnet-tfhub_frozen-backbone-trainable-head_trainable-mlp_classification-head-only/split_seed_42"
ORIGINAL = {
    "icbhi_to_icbhi": "ficheros/predicciones_originales/modelos/04-yamnet-trainable-head-confirm/icbhi-to-icbhi/{h}/run_20260527_072751",
    "icbhi_to_sprsound": "ficheros/predicciones_originales/modelos/04-yamnet-trainable-head-confirm/icbhi-to-sprsound/{h}/run_20260527_072855",
    "sprsound_to_sprsound": "ficheros/predicciones_originales/modelos/04-yamnet-trainable-head/sprsound-to-sprsound/{h}/run_20260526_221513",
    "sprsound_to_icbhi": "ficheros/predicciones_originales/modelos/04-yamnet-trainable-head/sprsound-to-icbhi/{h}/run_20260526_221656",
}
VARIANTS = ["zero", "loop", "noise", "min0975"]


def best_f1_threshold(y, p):
    # Misma regla que select_thresholds_from_validation: candidatos = {0, 0.5, 1} U scores, primer maximo.
    best_f1, best_thr = -1.0, 0.5
    for thr in np.unique(np.concatenate(([0.0, 0.5, 1.0], p))):
        f1 = f1_score(y, (p >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


def load_run(run_dir, split):
    frames = []
    for seed_dir in sorted(Path(run_dir).glob("seed_*")):
        df = pd.read_csv(seed_dir / "predictions" / f"predicciones_{split}.csv")
        frames.append(df.set_index("sample_path")[["y_true", "prob_adventitious"]].rename(
            columns={"prob_adventitious": seed_dir.name}))
    base = frames[0]
    for f in frames[1:]:
        assert f.index.equals(base.index) and (f["y_true"] == base["y_true"]).all()
    probs = pd.concat([f.drop(columns="y_true") for f in frames], axis=1)
    return base["y_true"], probs


def unpadded_test_paths(protocol):
    # Eventos de test >=0.975 s del dataset destino (columna 'padded' de la reextraccion loop).
    target = protocol.split("_to_")[1]
    meta = pd.read_csv(f"ficheros/3 Embeddings YAMNet/r1_20261029_{target}_seed42_yamnet_looppad_mean_pool_1024/"
                       "segmentado_test_metadata.csv")
    return set(meta.loc[~meta["padded"].astype(bool), "sample_path"])


def ensemble_row(run_dir, protocol, variant, source, test_subset="all"):
    yv, pv = load_run(run_dir, "validation")
    yt, pt = load_run(run_dir, "test")
    if test_subset == "min0975":
        keep = yt.index.isin(list(unpadded_test_paths(protocol)))
        yt, pt = yt[keep], pt[keep]
    ev, et = pv.mean(axis=1).to_numpy(), pt.mean(axis=1).to_numpy()
    thr = best_f1_threshold(yv.to_numpy(), ev)
    pred = (et >= thr).astype(int)
    seed_aurocs = [roc_auc_score(yt, pt[c]) for c in pt.columns]
    return {
        "protocol": protocol, "variant": variant, "source": source, "test_subset": test_subset, "n_seeds": pt.shape[1],
        "n_test": len(yt), "test_prevalence": float(yt.mean()),
        "auroc": roc_auc_score(yt, et), "auprc": average_precision_score(yt, et),
        "val_threshold": thr, "f1": f1_score(yt, pred, zero_division=0),
        "sensitivity": float(pred[yt.to_numpy() == 1].mean()),
        "specificity": float(1 - pred[yt.to_numpy() == 0].mean()),
        "auroc_seed_mean": float(np.mean(seed_aurocs)), "auroc_seed_sd": float(np.std(seed_aurocs, ddof=1)) if len(seed_aurocs) > 1 else np.nan,
        "run_dir": str(run_dir),
    }


rows = []
for test_subset in ("all", "min0975"):
    for protocol, template in ORIGINAL.items():
        rows.append(ensemble_row(template.format(h=HEAD), protocol, "zero", "original", test_subset))
        for variant in VARIANTS:
            run_dir = Path(f"modelos/r1-20261029-yamnet-head-padding-{variant}/{protocol.replace('_', '-')}/{HEAD}/r1_20261029_{variant}")
            if not run_dir.exists() or (variant == "min0975" and test_subset == "all"):
                continue
            # min0975 ya se evalua solo sobre eventos >=0.975 s; se etiqueta en ese bloque.
            rows.append(ensemble_row(run_dir, protocol, variant, "r1", "all" if variant == "min0975" else test_subset))
            if variant == "min0975":
                rows[-1]["test_subset"] = "min0975"
OUT_DIR.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "padding_ensemble_metrics.csv", index=False)
with pd.option_context("display.width", 200, "display.max_columns", 20):
    print(df.drop(columns="run_dir").round(4).to_string(index=False))
