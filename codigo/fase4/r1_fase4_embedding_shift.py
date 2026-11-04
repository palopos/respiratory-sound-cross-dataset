#!/usr/bin/env python3
"""R1: proyeccion UMAP y siluetas de los embeddings YAMNet de test (figura de dataset shift).

Copia de _prepare_embedding_table, _reduce_embeddings y del calculo de siluetas de
codigo/analysis/Analyze_Frozen_Baselines.py, con los mismos parametros
(PCA 50 + UMAP n_neighbors=25, min_dist=0.15, random_state=42; siluetas en PCA 30).

Modos: zero (comprobacion frente a embedding_shift_summary.csv del envio) y loop.
wav2vec 2.0 congelado no depende del relleno: se reutiliza la proyeccion del envio.

Uso: python codigo/fase4/r1_fase4_embedding_shift.py {zero|loop}
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

R1 = Path(__file__).resolve().parents[2]
EMB = R1 / "ficheros/3 Embeddings YAMNet"
INPUTS = {
    "zero": {
        "sprsound": EMB / "0 SPRSound_seed42_yamnet_tfhub_mean_pool_1024",
        "icbhi": EMB / "0 ICBHI_seed42_yamnet_tfhub_mean_pool_1024",
    },
    "loop": {
        "sprsound": EMB / "r1_20261029_sprsound_seed42_yamnet_looppad_mean_pool_1024",
        "icbhi": EMB / "r1_20261029_icbhi_seed42_yamnet_looppad_mean_pool_1024",
    },
}
OUT = R1 / "analisis/r1_20261101_fase4/embedding_shift"


def _prepare_embedding_table(inputs: dict) -> tuple[np.ndarray, pd.DataFrame]:
    frames = []
    embedding_arrays = []
    for dataset_key, base_dir in inputs.items():
        npz = np.load(base_dir / "segmentado_test_embeddings.npz", allow_pickle=False)
        embeddings = np.asarray(npz["embeddings"], dtype=np.float32)
        metadata = pd.read_csv(base_dir / "segmentado_test_metadata.csv")
        metadata["dataset_key"] = dataset_key
        if len(metadata) != len(embeddings):
            raise ValueError(f"Desajuste en embeddings/metadata para {base_dir}")
        frames.append(metadata)
        embedding_arrays.append(embeddings)
    return np.vstack(embedding_arrays), pd.concat(frames, ignore_index=True)


def _reduce_embeddings(embeddings: np.ndarray) -> np.ndarray:
    scaled = StandardScaler().fit_transform(embeddings)
    n_components = min(50, scaled.shape[0] - 1, scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    pca_embeddings = pca.fit_transform(scaled)
    reducer = umap.UMAP(n_components=2, n_neighbors=25, min_dist=0.15, metric="euclidean", random_state=42)
    return reducer.fit_transform(pca_embeddings)


def main() -> None:
    mode = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    embeddings, metadata = _prepare_embedding_table(INPUTS[mode])
    coords = _reduce_embeddings(embeddings)
    metadata["coord_1"] = coords[:, 0]
    metadata["coord_2"] = coords[:, 1]
    metadata.to_csv(OUT / f"yamnet_test_embedding_projection_{mode}.csv", index=False)

    scaled = StandardScaler().fit_transform(embeddings)
    pca_dim = min(30, scaled.shape[0] - 1, scaled.shape[1])
    pca_embeddings = PCA(n_components=pca_dim, random_state=42).fit_transform(scaled)
    dataset_codes = metadata["dataset_key"].astype("category").cat.codes
    class_codes = metadata["label_binary"].astype(int)
    summary = pd.DataFrame([{
        "model_family": "yamnet", "padding": mode, "projection_method": "umap", "n_segments": len(metadata),
        "dataset_silhouette": float(silhouette_score(pca_embeddings, dataset_codes)),
        "class_silhouette": float(silhouette_score(pca_embeddings, class_codes)),
    }])
    summary.to_csv(OUT / f"embedding_shift_summary_{mode}.csv", index=False)
    (OUT / f"manifest_{mode}.json").write_text(json.dumps({
        "script": "codigo/fase4/r1_fase4_embedding_shift.py", "mode": mode,
        "fecha": datetime.now().isoformat(timespec="seconds"), "umap_version": umap.__version__,
        "inputs": {k: str(Path(v).relative_to(R1)) for k, v in INPUTS[mode].items()},
    }, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
