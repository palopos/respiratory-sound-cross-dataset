#!/usr/bin/env python3
"""R1: distancias de representacion ICBHI train vs SPRSound test (figura de distancias).

Adaptado de codigo/analysis/Quantify_Domain_Distance.py (copia del script del
envio). Las funciones de submuestreo, proyeccion, MMD y sliced Wasserstein se copian
sin cambios, con las mismas constantes y semillas de muestreo.

Modos:
  zero  -> embeddings YAMNet con relleno de ceros y cabeza entrenable del envio
           (comprobacion: debe reproducir domain_distance_summary.csv del envio).
  loop  -> embeddings YAMNet con relleno loop y cabeza entrenable R1 (loop).

La CNN no depende del relleno de YAMNet: sus filas se toman de la tabla del envio
(ficheros/figuras_originales/domain_distance/domain_distance_raw.csv).

Uso: python codigo/fase4/r1_fase4_domain_distance.py {zero|loop}
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler

R1 = Path(__file__).resolve().parents[2]
ROOT = R1.parents[1]  # Fix_Paper1 (solo lectura)

ANALYSIS_SEED = 12345
MAX_SAMPLES_PER_GROUP = 1500
MAX_MEDIAN_HEURISTIC_SAMPLES = 1000
PCA_COMPONENTS = 32
SLICED_WASSERSTEIN_PROJECTIONS = 128

EMB = R1 / "ficheros/3 Embeddings YAMNet"
HEAD = "yamnet/yamnet_google-yamnet-tfhub_frozen-backbone-trainable-head_trainable-mlp_classification-head-only/split_seed_42"
CONFIG = {
    "zero": {
        "source_train": EMB / "0 ICBHI_seed42_yamnet_tfhub_mean_pool_1024",
        "target_test": EMB / "0 SPRSound_seed42_yamnet_tfhub_mean_pool_1024",
        "head_run": ROOT / "modelos/04-yamnet-trainable-head-confirm/icbhi-to-sprsound" / HEAD / "run_20260527_072855",
    },
    "loop": {
        "source_train": EMB / "r1_20261029_icbhi_seed42_yamnet_looppad_mean_pool_1024",
        "target_test": EMB / "r1_20261029_sprsound_seed42_yamnet_looppad_mean_pool_1024",
        "head_run": R1 / "modelos/r1-20261029-yamnet-head-padding-loop/icbhi-to-sprsound" / HEAD / "r1_20261029_loop",
    },
}
CNN_RAW = R1 / "ficheros/figuras_originales/domain_distance/domain_distance_raw.csv"
OUT = R1 / "analisis/r1_20261101_fase4/domain_distance"


def load_embedding_block(base_dir: Path, split_name: str) -> tuple[np.ndarray, pd.DataFrame]:
    payload = np.load(base_dir / f"{split_name}_embeddings.npz", allow_pickle=False)
    metadata = pd.read_csv(base_dir / f"{split_name}_metadata.csv")
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    if len(metadata) != len(embeddings):
        raise ValueError(f"Metadata and embeddings length mismatch for {base_dir} / {split_name}")
    return embeddings, metadata


def build_yamnet_feature_extractor(model_path: Path) -> tf.keras.Model:
    model = tf.keras.models.load_model(model_path, compile=False)
    feature_layer = model.get_layer("dense_2")
    return tf.keras.Model(inputs=model.input, outputs=feature_layer.output, name="yamnet_head_features")


def subsample_for_condition(source_x, source_y, target_x, target_y, condition: str, rng: np.random.Generator):
    if condition == "overall":
        source_mask = np.ones(len(source_y), dtype=bool)
        target_mask = np.ones(len(target_y), dtype=bool)
    elif condition == "healthy":
        source_mask = source_y == 0
        target_mask = target_y == 0
    elif condition == "adventitious":
        source_mask = source_y == 1
        target_mask = target_y == 1
    else:
        raise ValueError(f"Unsupported condition: {condition}")

    src = np.asarray(source_x[source_mask], dtype=np.float32)
    tgt = np.asarray(target_x[target_mask], dtype=np.float32)
    src_y = np.asarray(source_y[source_mask], dtype=np.int64)
    tgt_y = np.asarray(target_y[target_mask], dtype=np.int64)

    n = min(len(src), len(tgt), MAX_SAMPLES_PER_GROUP)
    if n < 2:
        raise ValueError(f"Not enough samples for condition={condition}: source={len(src)} target={len(tgt)}")

    if len(src) > n:
        src_idx = rng.choice(len(src), size=n, replace=False)
        src, src_y = src[src_idx], src_y[src_idx]
    if len(tgt) > n:
        tgt_idx = rng.choice(len(tgt), size=n, replace=False)
        tgt, tgt_y = tgt[tgt_idx], tgt_y[tgt_idx]
    return src, src_y, tgt, tgt_y


def project_features(source_x: np.ndarray, target_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pooled = np.concatenate([source_x, target_x], axis=0)
    scaler = StandardScaler()
    pooled = scaler.fit_transform(pooled)
    n_components = min(PCA_COMPONENTS, pooled.shape[1], pooled.shape[0] - 1)
    if n_components >= 2 and n_components < pooled.shape[1]:
        pca = PCA(n_components=n_components, random_state=ANALYSIS_SEED)
        pooled = pca.fit_transform(pooled)
    source_proj = pooled[: len(source_x)]
    target_proj = pooled[len(source_x) :]
    return source_proj.astype(np.float32), target_proj.astype(np.float32)


def estimate_gamma_median_heuristic(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> float:
    z = np.concatenate([x, y], axis=0)
    if len(z) > MAX_MEDIAN_HEURISTIC_SAMPLES:
        idx = rng.choice(len(z), size=MAX_MEDIAN_HEURISTIC_SAMPLES, replace=False)
        z = z[idx]
    sq_dists = pdist(z, metric="sqeuclidean")
    median_sq = float(np.median(sq_dists)) if len(sq_dists) else 1.0
    median_sq = max(median_sq, 1e-6)
    return 1.0 / (2.0 * median_sq)


def rbf_mmd(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> float:
    gamma = estimate_gamma_median_heuristic(x, y, rng)
    k_xx = rbf_kernel(x, x, gamma=gamma)
    k_yy = rbf_kernel(y, y, gamma=gamma)
    k_xy = rbf_kernel(x, y, gamma=gamma)
    n = len(x)
    m = len(y)
    xx = (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
    yy = (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
    xy = k_xy.mean()
    mmd2 = float(xx + yy - 2.0 * xy)
    return math.sqrt(max(mmd2, 0.0))


def sliced_wasserstein(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_projections: int = SLICED_WASSERSTEIN_PROJECTIONS) -> float:
    dim = x.shape[1]
    directions = rng.normal(size=(n_projections, dim)).astype(np.float32)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.clip(norms, 1e-12, None)
    x_proj = x @ directions.T
    y_proj = y @ directions.T
    distances = [
        wasserstein_distance(x_proj[:, i], y_proj[:, i])
        for i in range(n_projections)
    ]
    return float(np.mean(distances))


def compute_condition_metrics(source_x, source_y, target_x, target_y, condition: str, seed_for_sampling: int):
    rng = np.random.default_rng(seed_for_sampling)
    src, src_y, tgt, tgt_y = subsample_for_condition(source_x, source_y, target_x, target_y, condition, rng)
    src_proj, tgt_proj = project_features(src, tgt)
    pooled = np.concatenate([src_proj, tgt_proj], axis=0)
    domain_codes = np.concatenate([np.zeros(len(src_proj), dtype=int), np.ones(len(tgt_proj), dtype=int)])
    class_codes = np.concatenate([src_y, tgt_y], axis=0)

    metrics = {
        "n_source": int(len(src_proj)),
        "n_target": int(len(tgt_proj)),
        "rbf_mmd": rbf_mmd(src_proj, tgt_proj, rng),
        "sliced_wasserstein": sliced_wasserstein(src_proj, tgt_proj, rng),
        "domain_centroid_distance": float(np.linalg.norm(src_proj.mean(axis=0) - tgt_proj.mean(axis=0))),
    }
    if condition == "overall" and len(np.unique(domain_codes)) == 2 and len(np.unique(class_codes)) == 2:
        metrics["domain_silhouette"] = float(silhouette_score(pooled, domain_codes))
        metrics["class_silhouette"] = float(silhouette_score(pooled, class_codes))
    else:
        metrics["domain_silhouette"] = np.nan
        metrics["class_silhouette"] = np.nan
    return metrics


def collect_representations(cfg: dict) -> list[dict]:
    source_x, source_meta = load_embedding_block(cfg["source_train"], "segmentado_train")
    target_x, target_meta = load_embedding_block(cfg["target_test"], "segmentado_test")
    items = [{
        "representation": "yamnet_frozen_embedding", "seed": 42,
        "source": {"x": source_x, "meta": source_meta}, "target": {"x": target_x, "meta": target_meta},
    }]
    for seed_dir in sorted(cfg["head_run"].glob("seed_*")):
        seed = int(seed_dir.name.replace("seed_", ""))
        feature_model = build_yamnet_feature_extractor(seed_dir / "checkpoints" / "best_model.keras")
        items.append({
            "representation": "yamnet_trainable_head", "seed": seed,
            "source": {"x": feature_model.predict(source_x, batch_size=512, verbose=0).astype(np.float32), "meta": source_meta},
            "target": {"x": feature_model.predict(target_x, batch_size=512, verbose=0).astype(np.float32), "meta": target_meta},
        })
    return items


def quantify(items: list[dict]) -> pd.DataFrame:
    rows = []
    for item in items:
        seed = item["seed"]
        source_x = np.asarray(item["source"]["x"], dtype=np.float32)
        target_x = np.asarray(item["target"]["x"], dtype=np.float32)
        source_y = np.asarray(item["source"]["meta"]["label_binary"], dtype=np.int64)
        target_y = np.asarray(item["target"]["meta"]["label_binary"], dtype=np.int64)
        for condition_idx, condition in enumerate(("overall", "healthy", "adventitious")):
            metrics = compute_condition_metrics(
                source_x, source_y, target_x, target_y, condition,
                seed_for_sampling=ANALYSIS_SEED + seed * 100 + condition_idx,
            )
            rows.append({"representation": item["representation"], "seed": seed,
                         "comparison": "icbhi_train_vs_sprsound_test", "condition": condition, **metrics})
    return pd.DataFrame(rows)


def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["representation", "comparison", "condition"], as_index=False)
        .agg(
            rbf_mmd_mean=("rbf_mmd", "mean"),
            rbf_mmd_std=("rbf_mmd", "std"),
            sliced_wasserstein_mean=("sliced_wasserstein", "mean"),
            sliced_wasserstein_std=("sliced_wasserstein", "std"),
            domain_centroid_distance_mean=("domain_centroid_distance", "mean"),
            domain_centroid_distance_std=("domain_centroid_distance", "std"),
            domain_silhouette_mean=("domain_silhouette", "mean"),
            class_silhouette_mean=("class_silhouette", "mean"),
            n_source_mean=("n_source", "mean"),
            n_target_mean=("n_target", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )
    for column in summary.columns:
        if column.endswith("_std"):
            summary[column] = summary[column].fillna(0.0)
    return summary


def main() -> None:
    mode = sys.argv[1]
    cfg = CONFIG[mode]
    OUT.mkdir(parents=True, exist_ok=True)
    raw = quantify(collect_representations(cfg))
    cnn = pd.read_csv(CNN_RAW)
    raw = pd.concat([cnn[cnn["representation"] == "cnn_penultimate"], raw], ignore_index=True)
    raw.to_csv(OUT / f"domain_distance_raw_{mode}.csv", index=False)
    summary = aggregate_metrics(raw)
    summary.to_csv(OUT / f"domain_distance_summary_{mode}.csv", index=False)
    manifest = {
        "script": "codigo/fase4/r1_fase4_domain_distance.py", "mode": mode,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "analysis_seed": ANALYSIS_SEED, "pca_components": PCA_COMPONENTS,
        "max_samples_per_group": MAX_SAMPLES_PER_GROUP,
        "inputs": {k: str(Path(v).relative_to(R1)) for k, v in cfg.items()}, "cnn_rows_from": str(CNN_RAW.relative_to(R1)),
    }
    (OUT / f"manifest_{mode}.json").write_text(json.dumps(manifest, indent=2))
    print(summary[["representation", "condition", "rbf_mmd_mean", "sliced_wasserstein_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
