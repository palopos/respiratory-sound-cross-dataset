"""R1 Fase 1: subconjunto de embeddings YAMNet originales restringido a eventos >=0.975 s (sin relleno).

La duracion original se toma de los metadatos de la reextraccion loop (columna 'padded'); para
esos eventos el embedding original no tiene relleno y es identico al de cualquier variante.
Uso: python codigo/preprocess/yamnet/r1_Build_YAMNet_Min0975.py  (desde la raiz del repositorio)
"""
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("ficheros/3 Embeddings YAMNet")
DATASETS = {"icbhi": "0 ICBHI", "sprsound": "0 SPRSound"}

for label, name in DATASETS.items():
    src = BASE / f"{name}_seed42_yamnet_tfhub_mean_pool_1024"
    durations = BASE / f"r1_20261029_{label}_seed42_yamnet_looppad_mean_pool_1024"
    out = BASE / f"r1_20261029_{label}_seed42_yamnet_min0975_mean_pool_1024"
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ("train", "validation", "test"):
        payload = np.load(src / f"segmentado_{split}_embeddings.npz")
        meta = pd.read_csv(src / f"segmentado_{split}_metadata.csv")
        dur = pd.read_csv(durations / f"segmentado_{split}_metadata.csv")
        assert (meta["sample_path"].values == dur["sample_path"].values).all()
        keep = ~dur["padded"].to_numpy(dtype=bool)
        meta = meta.loc[keep].assign(original_duration_seconds=dur.loc[keep, "original_duration_seconds"].values)
        np.savez_compressed(out / f"segmentado_{split}_embeddings.npz",
                            embeddings=payload["embeddings"][keep], labels=payload["labels"][keep])
        meta.to_csv(out / f"segmentado_{split}_metadata.csv", index=False)
        labels = payload["labels"][keep]
        summary[split] = {"n_kept": int(keep.sum()), "n_dropped": int((~keep).sum()),
                          "n_negative": int((labels == 0).sum()), "n_positive": int((labels == 1).sum())}
        print(label, split, summary[split])
    with open(out / "embedding_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"created_at": datetime.now().isoformat(timespec="seconds"), "source_embeddings": str(src),
                   "duration_source": str(durations), "rule": "original_duration_seconds >= 0.975 (padded == False)",
                   "splits": summary}, f, ensure_ascii=False, indent=2)
