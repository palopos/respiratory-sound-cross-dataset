"""R1 Fase 3 (R3.4, R2.4): SPRSound restringido a las etiquetas compartidas con ICBHI (sin Rhonchi ni Stridor).

Particion original (semilla 42). Se eliminan los eventos Rhonchi y Stridor de train, validation y test de SPRSound.
  - YAMNet (relleno loop): ficheros/3 Embeddings YAMNet/r1_20261029_sprsound_seed42_yamnet_looppad_commonlabels/
  - CNN: ficheros/2 Espectrogramas/r1_comunes/0 SPRSound_seed42_commonlabels_resize_dinamico_MEL64_SIZE64/npy/...
         (enlaces a los espectrogramas originales) y el test ICBHI original enlazado en
         ficheros/2 Espectrogramas/r1_comunes/0 ICBHI_seed42_resize_dinamico_MEL64_SIZE64/npy/segmentado_test
Uso: python codigo/fase3/r1_make_common_labels.py   (desde la raiz del repositorio)
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

EXCLUDE = {"Rhonchi", "Stridor"}
ORIG = Path("./ficheros/2 Espectrogramas")
SPEC_DST = Path("ficheros/2 Espectrogramas/r1_comunes")
EMB_SRC = Path("ficheros/3 Embeddings YAMNet/r1_20261029_sprsound_seed42_yamnet_looppad_mean_pool_1024")
EMB_DST = Path("ficheros/3 Embeddings YAMNet/r1_20261029_sprsound_seed42_yamnet_looppad_commonlabels")


def link_tree(src_npy, dst_npy, splits, exclude):
    counts = {}
    for split in splits:
        for class_dir in sorted((src_npy / f"segmentado_{split}").iterdir()):
            if not class_dir.is_dir() or class_dir.name in exclude:
                continue
            target = dst_npy / f"segmentado_{split}" / class_dir.name
            target.mkdir(parents=True, exist_ok=False)
            n = 0
            for f in sorted(class_dir.glob("*.npy")):
                os.symlink(f, target / f.name)
                n += 1
            counts[f"{split}/{class_dir.name}"] = n
    return counts


summary = {"excluded_labels": sorted(EXCLUDE)}
summary["cnn_sprsound"] = link_tree(ORIG / "0 SPRSound_seed42_resize_dinamico_MEL64_SIZE64" / "npy",
                                    SPEC_DST / "0 SPRSound_seed42_commonlabels_resize_dinamico_MEL64_SIZE64" / "npy",
                                    ("train", "validation", "test"), EXCLUDE)
summary["cnn_icbhi_test"] = link_tree(ORIG / "0 ICBHI_seed42_resize_dinamico_MEL64_SIZE64" / "npy",
                                      SPEC_DST / "0 ICBHI_seed42_resize_dinamico_MEL64_SIZE64" / "npy", ("test",), set())
EMB_DST.mkdir(parents=True, exist_ok=False)
summary["yamnet"] = {}
for split in ("train", "validation", "test"):
    payload = np.load(EMB_SRC / f"segmentado_{split}_embeddings.npz")
    meta = pd.read_csv(EMB_SRC / f"segmentado_{split}_metadata.csv")
    keep = ~meta["label_name"].isin(EXCLUDE).to_numpy()
    np.savez_compressed(EMB_DST / f"segmentado_{split}_embeddings.npz",
                        embeddings=payload["embeddings"][keep], labels=payload["labels"][keep])
    meta.loc[keep].to_csv(EMB_DST / f"segmentado_{split}_metadata.csv", index=False)
    summary["yamnet"][split] = {"kept": int(keep.sum()), "dropped": int((~keep).sum()),
                                "n_positive": int(payload["labels"][keep].sum())}
with open(EMB_DST / "embedding_manifest.json", "w", encoding="utf-8") as f:
    json.dump({"source": str(EMB_SRC), **summary}, f, indent=2)
print(json.dumps(summary, indent=1))
