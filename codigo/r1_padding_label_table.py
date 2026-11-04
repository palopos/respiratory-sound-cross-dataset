"""R1 Fase 1 / R4.CS5: proporcion de eventos con relleno (<0.975 s) y prevalencia de adventicios
en eventos con y sin relleno, por dataset y particion.

Uso: python codigo/r1_padding_label_table.py  (desde la raiz del repositorio)
Salida: analisis/r1_20261029_padding/padding_by_label.csv
"""
from pathlib import Path

import pandas as pd

BASE = Path("ficheros/3 Embeddings YAMNet")
OUT_DIR = Path("analisis/r1_20261029_padding")
rows = []
for label in ("icbhi", "sprsound"):
    for split in ("train", "validation", "test"):
        m = pd.read_csv(BASE / f"r1_20261029_{label}_seed42_yamnet_looppad_mean_pool_1024/segmentado_{split}_metadata.csv")
        pad, pos = m["padded"].astype(bool), m["label_binary"] == 1
        rows.append({
            "dataset": label, "split": split, "n_events": len(m),
            "n_padded": int(pad.sum()), "pct_padded": 100 * pad.mean(),
            "pct_padded_normal": 100 * pad[~pos].mean(), "pct_padded_adventitious": 100 * pad[pos].mean(),
            "prevalence_padded": pos[pad].mean(), "prevalence_unpadded": pos[~pad].mean(),
            "median_duration_s": m["original_duration_seconds"].median(),
        })
OUT_DIR.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "padding_by_label.csv", index=False)
print(df.round(3).to_string(index=False))
