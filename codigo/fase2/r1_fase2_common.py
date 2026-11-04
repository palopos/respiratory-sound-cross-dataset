"""R1 Fase 2: utilidades comunes (registro de modelos, ensembles, umbral de validacion, metadatos).

Todo se calcula desde predicciones guardadas; no se reentrena nada.
Los scripts de la fase se ejecutan desde la raiz del repositorio.
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ORIG = Path("ficheros/predicciones_originales")
R1_RUNS = Path("modelos")
EMB = Path("ficheros/3 Embeddings YAMNet")
SPLITS = {"icbhi": Path("ficheros/splits/0 ICBHI/splits_patient_independent_seed42.csv"),
          "sprsound": Path("ficheros/splits/0 SPRSound/splits_patient_independent_seed42.csv")}
EXPECTED_TEST = {"icbhi": 1414, "sprsound": 2991}
PROTOCOL_LABEL = {"sprsound_to_sprsound": "SPRSound -> SPRSound", "icbhi_to_icbhi": "ICBHI -> ICBHI",
                  "icbhi_to_sprsound": "ICBHI -> SPRSound", "sprsound_to_icbhi": "SPRSound -> ICBHI"}
W2V_SETUP = "wav2vec2/wav2vec2_facebook-wav2vec2-base_finetuned_sequence-classification-head_full/split_seed_42"
W2V_FIXED = {  # Fase 3: wav2vec 2.0 con el ajuste corregido (3 semillas); intra = mismos checkpoints en el test de origen
    "sprsound_to_icbhi": f"r1-20261029-wav2vec2-finetune-fixed/sprsound-to-icbhi/{W2V_SETUP}/r1_20261029_fixed",
    "sprsound_to_sprsound": f"r1-20261029-wav2vec2-finetune-fixed/sprsound-to-sprsound/{W2V_SETUP}/r1_20261029_fixed_sprsoundtest",
    "icbhi_to_sprsound": f"r1-20261030-wav2vec2-finetune-fixed/icbhi-to-sprsound/{W2V_SETUP}/r1_20261030_fixed",
    "icbhi_to_icbhi": f"r1-20261030-wav2vec2-finetune-fixed/icbhi-to-icbhi/{W2V_SETUP}/r1_20261030_fixed_icbhitest",
}
YAMNET_HEAD = ("yamnet/yamnet_google-yamnet-tfhub_frozen-backbone-trainable-head_trainable-mlp_"
               "classification-head-only/split_seed_42")


def _original_registry():
    rows = []
    audited = pd.read_csv(ORIG / "audited_model_comparison.csv")
    inv = {v: k for k, v in PROTOCOL_LABEL.items()}
    for _, r in audited.iterrows():
        rows.append({"protocol": inv[r["protocol"]], "model": r["model"], "evidence": r["evidence"],
                     "group": "paper", "run_dir": ORIG / r["source_run"]})
    rows.append({"protocol": "icbhi_to_sprsound", "model": "YAMNet trainable head + Mixup",
                 "evidence": "three-seed ensemble", "group": "paper",
                 "run_dir": ORIG / "modelos/10-yamnet-mixup-confirm/icbhi-to-sprsound/yamnet/yamnet_google-yamnet-tfhub_"
                 "frozen-backbone-trainable-head-mixup_trainable-mlp_classification-head-only/split_seed_42/run_20260527_085309"})
    rows.append({"protocol": "icbhi_to_sprsound", "model": "YAMNet trainable head + CORAL",
                 "evidence": "single-seed screening", "group": "paper",
                 "run_dir": ORIG / "modelos/07-yamnet-coral-screening/icbhi-to-sprsound/yamnet/yamnet_google-yamnet-tfhub_"
                 "coral-frozen-backbone-trainable-head_trainable-mlp_classification-head-only/split_seed_42/run_20260527_084024"})
    return rows


def _r1_registry():
    rows = []
    names = {"zero": "YAMNet trainable head (zero pad, R1 3 seeds)", "loop": "YAMNet trainable head (loop pad)",
             "noise": "YAMNet trainable head (noise pad)", "min0975": "YAMNet trainable head (events >=0.975 s)"}
    for variant, name in names.items():
        for protocol in PROTOCOL_LABEL:
            if variant == "zero" and source_of(protocol) == "icbhi":
                continue  # identico bit a bit a las ejecuciones confirm originales (comprobado en Fase 1)
            run_dir = (R1_RUNS / f"r1-20261029-yamnet-head-padding-{variant}" / protocol.replace("_", "-")
                       / YAMNET_HEAD / f"r1_20261029_{variant}")
            if run_dir.exists():
                rows.append({"protocol": protocol, "model": name, "evidence": "three-seed ensemble",
                             "group": f"r1_padding_{variant}", "run_dir": run_dir})
    for protocol, rel in W2V_FIXED.items():
        if (R1_RUNS / rel).exists():
            rows.append({"protocol": protocol, "model": "wav2vec 2.0 fine-tuned (R1 fixed)", "evidence": "three-seed ensemble",
                         "group": "r1_wav2vec2_fixed", "run_dir": R1_RUNS / rel})
    return rows


def model_registry(include_r1=True):
    rows = _original_registry() + (_r1_registry() if include_r1 else [])
    df = pd.DataFrame(rows)
    df["model_id"] = [f"{p}|{m}" for p, m in zip(df["protocol"], df["model"])]
    return df


def segment_stem(sample_path):
    return Path(str(sample_path)).stem


def recording_of(stem):
    return re.sub(r"_ev\d+$", "", stem)


def _load_split(run_dir, split):
    seed_dirs = sorted(Path(run_dir).glob("seed_*"))
    if not seed_dirs:
        raise FileNotFoundError(f"Sin carpetas seed_* en {run_dir}")
    merged = None
    for seed_dir in seed_dirs:
        df = pd.read_csv(seed_dir / "predictions" / f"predicciones_{split}.csv")
        df = pd.DataFrame({"stem": df["sample_path"].map(segment_stem), "y_true": df["y_true"].astype(int),
                           seed_dir.name: df["prob_adventitious"].astype(float)})
        if df["stem"].duplicated().any():
            raise ValueError(f"Segmentos duplicados en {seed_dir}/{split}")
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=["stem", "y_true"], how="inner", validate="one_to_one")
            if len(merged) != len(df):
                raise ValueError(f"Semillas no alineadas en {run_dir}/{split}")
    seed_cols = [c for c in merged.columns if c.startswith("seed_")]
    merged["prob"] = merged[seed_cols].mean(axis=1)
    merged["n_seeds"] = len(seed_cols)
    return merged


def val_best_f1_threshold(y, p):
    """Misma regla que select_thresholds_from_validation de los scripts de entrenamiento:
    candidatos = {0, 0.5, 1} U scores (ordenados), se queda el primer maximo de F1."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], p)))
    # F1 vectorizado para todos los candidatos: predice positivo si p >= thr.
    order = np.argsort(-p, kind="mergesort")
    p_sorted, y_sorted = p[order], y[order]
    tp_cum = np.concatenate(([0], np.cumsum(y_sorted)))
    n_pos = y.sum()
    # numero de scores >= thr para cada candidato
    n_pred = len(p) - np.searchsorted(np.sort(p), candidates, side="left")
    tp = tp_cum[n_pred]
    f1 = np.where(n_pred + n_pos > 0, 2 * tp / np.maximum(n_pred + n_pos, 1), 0.0)
    return float(candidates[int(np.argmax(f1))])


def load_ensemble(run_dir):
    """Devuelve (validacion, test) con prob = media de semillas y el umbral val_best_f1."""
    val, test = _load_split(run_dir, "validation"), _load_split(run_dir, "test")
    thr = val_best_f1_threshold(val["y_true"], val["prob"])
    return val, test, thr


def split_metadata(dataset):
    s = pd.read_csv(SPLITS[dataset])
    keep = ["recording_id", "split", "patient_group"] + [c for c in ("equipment", "chest_location", "acquisition_mode",
                                                                       "age", "gender", "location") if c in s.columns]
    return s[keep].rename(columns={"patient_group": "patient_id"})


def event_metadata(dataset, split):
    """Duracion original y etiqueta detallada por segmento (de la reextraccion loop de la Fase 1)."""
    m = pd.read_csv(EMB / f"r1_20261029_{dataset}_seed42_yamnet_looppad_mean_pool_1024" / f"segmentado_{split}_metadata.csv")
    return pd.DataFrame({"stem": m["sample_path"].map(segment_stem), "label_name": m["label_name"],
                         "duration_s": m["original_duration_seconds"], "padded": m["padded"].astype(bool)})


def attach_metadata(df, dataset, split):
    out = df.copy()
    out["recording_id"] = out["stem"].map(recording_of)
    out = out.merge(split_metadata(dataset), on="recording_id", how="left", validate="many_to_one")
    if out["patient_id"].isna().any():
        missing = out.loc[out["patient_id"].isna(), "recording_id"].unique()[:5]
        raise ValueError(f"Segmentos sin paciente en {dataset}/{split}: {missing}")
    if (out["split"] != split).any():
        raise ValueError(f"Segmentos de {dataset} asignados a una particion distinta de {split}")
    out = out.merge(event_metadata(dataset, split), on="stem", how="left", validate="one_to_one")
    if out["duration_s"].isna().any():
        raise ValueError(f"Segmentos sin duracion en {dataset}/{split}")
    return out


def target_of(protocol):
    return protocol.split("_to_")[1]


def source_of(protocol):
    return protocol.split("_to_")[0]


def point_metrics(y, p, thr):
    y = np.asarray(y, dtype=int)
    pred = (np.asarray(p) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
            "f1": f1_score(y, pred, zero_division=0),
            "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "ppv": tp / (tp + fp) if tp + fp else np.nan, "npv": tn / (tn + fn) if tn + fn else np.nan,
            "pred_positive_rate": float(pred.mean()), "tn": tn, "fp": fp, "fn": fn, "tp": tp}
