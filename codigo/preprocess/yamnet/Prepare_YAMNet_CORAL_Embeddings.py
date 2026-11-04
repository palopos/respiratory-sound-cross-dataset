from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path


def _find_project_root(start_dir: str) -> Path:
    current_dir = Path(start_dir).resolve()
    while True:
        if (current_dir / "environment.yml").exists() and (current_dir / "ficheros").is_dir():
            return current_dir
        if current_dir.parent == current_dir:
            raise FileNotFoundError("Could not locate the Fix_Paper1 project root.")
        current_dir = current_dir.parent


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _find_project_root(str(SCRIPT_DIR))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd


SOURCE_DATASET_LABEL = os.environ.get("SOURCE_DATASET_LABEL", "icbhi")
TARGET_DATASET_LABEL = os.environ.get("TARGET_DATASET_LABEL", "sprsound")
SOURCE_EMBEDDINGS_DIR = Path(
    os.environ.get(
        "SOURCE_EMBEDDINGS_DIR",
        "./ficheros/3 Embeddings YAMNet/0 ICBHI_seed42_yamnet_tfhub_mean_pool_1024",
    )
)
TARGET_EMBEDDINGS_DIR = Path(
    os.environ.get(
        "TARGET_EMBEDDINGS_DIR",
        "./ficheros/3 Embeddings YAMNet/0 SPRSound_seed42_yamnet_tfhub_mean_pool_1024",
    )
)
OUTPUT_DIR = Path(
    os.environ.get(
        "OUTPUT_DIR",
        f"./ficheros/4 Domain Adaptation/YAMNet_CORAL_{SOURCE_DATASET_LABEL}_to_{TARGET_DATASET_LABEL}_seed42",
    )
)
TARGET_REFERENCE_SPLITS = [
    item.strip().lower()
    for item in os.environ.get("TARGET_REFERENCE_SPLITS", "train").replace(";", ",").split(",")
    if item.strip()
]
SPLIT_SEED = int(os.environ.get("SPLIT_SEED", 42))
CORAL_EPSILON = float(os.environ.get("CORAL_EPSILON", 1e-3))


def load_npz_and_metadata(base_dir: Path, split_name: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    npz_path = base_dir / f"segmentado_{split_name}_embeddings.npz"
    metadata_path = base_dir / f"segmentado_{split_name}_metadata.csv"
    if not npz_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Missing embedding artifacts for split '{split_name}' under {base_dir}")
    payload = np.load(npz_path, allow_pickle=False)
    embeddings = np.asarray(payload["embeddings"], dtype=np.float64)
    labels = np.asarray(payload["labels"], dtype=np.int64)
    metadata = pd.read_csv(metadata_path)
    if len(embeddings) != len(labels) or len(embeddings) != len(metadata):
        raise ValueError(
            f"Mismatch in {npz_path}: embeddings={len(embeddings)} labels={len(labels)} metadata={len(metadata)}"
        )
    return embeddings, labels, metadata


def covariance_sqrt(matrix: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, 1e-12, None)
    return (eigvecs * np.sqrt(eigvals)) @ eigvecs.T


def covariance_inv_sqrt(matrix: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, 1e-12, None)
    return (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T


def fit_coral_transform(source_train_x: np.ndarray, target_reference_x: np.ndarray) -> dict[str, np.ndarray]:
    source_mean = source_train_x.mean(axis=0, keepdims=True)
    target_mean = target_reference_x.mean(axis=0, keepdims=True)
    source_centered = source_train_x - source_mean
    target_centered = target_reference_x - target_mean

    feature_dim = source_train_x.shape[1]
    identity = np.eye(feature_dim, dtype=np.float64)
    source_cov = np.cov(source_centered, rowvar=False) + CORAL_EPSILON * identity
    target_cov = np.cov(target_centered, rowvar=False) + CORAL_EPSILON * identity

    transform = covariance_inv_sqrt(source_cov) @ covariance_sqrt(target_cov)
    return {
        "source_mean": source_mean,
        "target_mean": target_mean,
        "transform": transform,
    }


def apply_coral_transform(features: np.ndarray, coral: dict[str, np.ndarray]) -> np.ndarray:
    centered = features - coral["source_mean"]
    adapted = centered @ coral["transform"] + coral["target_mean"]
    return np.asarray(adapted, dtype=np.float32)


def save_split(output_dir: Path, split_name: str, embeddings: np.ndarray, labels: np.ndarray, metadata: pd.DataFrame) -> None:
    np.savez_compressed(
        output_dir / f"segmentado_{split_name}_embeddings.npz",
        embeddings=np.asarray(embeddings, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
    )
    metadata.to_csv(output_dir / f"segmentado_{split_name}_metadata.csv", index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_train_x, source_train_y, source_train_meta = load_npz_and_metadata(SOURCE_EMBEDDINGS_DIR, "train")
    source_val_x, source_val_y, source_val_meta = load_npz_and_metadata(SOURCE_EMBEDDINGS_DIR, "validation")

    target_reference_embeddings = []
    target_reference_rows = 0
    for split_name in TARGET_REFERENCE_SPLITS:
        split_x, _, split_meta = load_npz_and_metadata(TARGET_EMBEDDINGS_DIR, split_name)
        target_reference_embeddings.append(split_x)
        target_reference_rows += len(split_meta)
    if not target_reference_embeddings:
        raise ValueError("TARGET_REFERENCE_SPLITS does not contain any valid split names.")
    target_reference_x = np.concatenate(target_reference_embeddings, axis=0)

    coral = fit_coral_transform(source_train_x, target_reference_x)
    adapted_train_x = apply_coral_transform(source_train_x, coral)
    adapted_val_x = apply_coral_transform(source_val_x, coral)

    save_split(OUTPUT_DIR, "train", adapted_train_x, source_train_y, source_train_meta)
    save_split(OUTPUT_DIR, "validation", adapted_val_x, source_val_y, source_val_meta)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "CORAL",
        "split_seed": SPLIT_SEED,
        "source_dataset_label": SOURCE_DATASET_LABEL,
        "target_dataset_label": TARGET_DATASET_LABEL,
        "source_embeddings_dir": str(SOURCE_EMBEDDINGS_DIR),
        "target_embeddings_dir": str(TARGET_EMBEDDINGS_DIR),
        "output_dir": str(OUTPUT_DIR),
        "target_reference_splits": TARGET_REFERENCE_SPLITS,
        "target_reference_rows": int(target_reference_rows),
        "coral_epsilon": CORAL_EPSILON,
        "embedding_dim": int(source_train_x.shape[1]),
        "notes": (
            "CORAL adaptation fitted on source-train embeddings and unlabeled target-reference embeddings. "
            "Only source train/validation are transformed. Target test remains untouched."
        ),
        "generated_files": {
            "train_npz": str(OUTPUT_DIR / "segmentado_train_embeddings.npz"),
            "train_metadata": str(OUTPUT_DIR / "segmentado_train_metadata.csv"),
            "validation_npz": str(OUTPUT_DIR / "segmentado_validation_embeddings.npz"),
            "validation_metadata": str(OUTPUT_DIR / "segmentado_validation_metadata.csv"),
        },
    }
    with open(OUTPUT_DIR / "coral_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=True, indent=2)

    print(f">>> CORAL-adapted embeddings saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
