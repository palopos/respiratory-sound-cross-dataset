"""R1 Fase 3 (R2.3, R3.2): particiones por paciente alternativas con nuevas semillas de particion.

Reproduce el algoritmo de asignacion de los scripts CNN originales sin releer el audio crudo:
  - ICBHI: allocate_patient_groups_by_recording_fraction (60/20/20 por numero de grabaciones,
    estratificado por paciente con/sin algun evento adventicio).
  - SPRSound: split_train_validation_by_patient (20 % de validacion); el test oficial no cambia.
El orden de entrada es el mismo que en los scripts (grabaciones ordenadas por recording_id).
Comprobacion: con la semilla 42 debe reproducir exactamente splits_patient_independent_seed42.csv.

Para cada semilla nueva escribe:
  ficheros/splits/0 <DS>/splits_patient_independent_seed<S>.csv
  ficheros/2 Espectrogramas/r1_particiones/0 <DS>_seed<S>_resize_dinamico_MEL64_SIZE64/npy/segmentado_<split>/<clase>/*.npy
      (enlaces simbolicos fichero a fichero a los espectrogramas originales; no se recalcula nada)
  ficheros/3 Embeddings YAMNet/r1_particiones/r1_<ds>_seed<S>_yamnet_looppad_mean_pool_1024/  (npz reordenados, copia)
Uso: python codigo/fase3/r1_make_partitions.py   (desde la raiz del repositorio)
"""
from collections import defaultdict
import json
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd

NEW_SEEDS = [101, 202, 303]
DATASETS = {"icbhi": "0 ICBHI", "sprsound": "0 SPRSound"}
SPLITS_DIR = Path("ficheros/splits")
SPEC_SRC = Path("ficheros/2 Espectrogramas")
SPEC_DST = Path("ficheros/2 Espectrogramas/r1_particiones")
EMB_SRC = Path("ficheros/3 Embeddings YAMNet")
EMB_DST = Path("ficheros/3 Embeddings YAMNet/r1_particiones")
ORIGINAL_SPEC_ROOT = Path("./ficheros/2 Espectrogramas")
SPLIT_NAMES = ("train", "validation", "test")


# --- Copia literal de la logica de asignacion de Train_CNN_Scratch_ICBHI.py ---
def allocate_patient_groups_by_recording_fraction(patient_groups, validation_fraction=0.20, test_fraction=0.20, seed=42):
    rng = random.Random(seed)
    positive_groups, negative_groups = [], []
    for group_records in patient_groups.values():
        has_adventitious = any(r["n_adventitious_events"] > 0 for r in group_records)
        if has_adventitious:
            positive_groups.append(group_records)
        else:
            negative_groups.append(group_records)

    assignments = {"train": [], "validation": [], "test": []}
    for bucket in (positive_groups, negative_groups):
        rng.shuffle(bucket)
        bucket_record_count = sum(len(group_records) for group_records in bucket)
        target_test = int(round(bucket_record_count * test_fraction))
        target_validation = int(round(bucket_record_count * validation_fraction))

        test_count = 0
        validation_count = 0
        for group_records in bucket:
            if test_count < target_test:
                assignments["test"].append(group_records)
                test_count += len(group_records)
            elif validation_count < target_validation:
                assignments["validation"].append(group_records)
                validation_count += len(group_records)
            else:
                assignments["train"].append(group_records)
    return {split: [r["recording_id"] for g in groups for r in g] for split, groups in assignments.items()}


# --- Copia literal de la logica de asignacion de Train_CNN_Scratch_SPRSound.py ---
def split_train_validation_by_patient(patient_groups, validation_fraction=0.20, seed=42):
    rng = random.Random(seed)
    positive_groups, negative_groups = [], []
    for group_records in patient_groups.values():
        has_adventitious = any(r["n_adventitious_events"] > 0 for r in group_records)
        if has_adventitious:
            positive_groups.append(group_records)
        else:
            negative_groups.append(group_records)

    validation_groups = []
    for bucket in (positive_groups, negative_groups):
        rng.shuffle(bucket)
        bucket_record_count = sum(len(group_records) for group_records in bucket)
        target_count = max(1, int(round(bucket_record_count * validation_fraction))) if bucket_record_count else 0
        selected_count = 0
        for group_records in bucket:
            if selected_count >= target_count and validation_groups:
                break
            validation_groups.append(group_records)
            selected_count += len(group_records)

    validation_patient_groups = {group_records[0]["patient_group"] for group_records in validation_groups}
    out = {"train": [], "validation": []}
    for patient_group, group_records in patient_groups.items():
        target = "validation" if patient_group in validation_patient_groups else "train"
        out[target].extend(r["recording_id"] for r in group_records)
    return out


def group_by_patient(records):
    groups = defaultdict(list)
    for r in sorted(records, key=lambda r: r["recording_id"]):
        groups[r["patient_group"]].append(r)
    return groups


def assign(dataset, base, seed):
    if dataset == "icbhi":
        result = allocate_patient_groups_by_recording_fraction(group_by_patient(base.to_dict("records")), seed=seed)
    else:
        pool = base[base["split"].isin(["train", "validation"])]
        result = split_train_validation_by_patient(group_by_patient(pool.to_dict("records")), seed=seed)
        result["test"] = base.loc[base["split"] == "test", "recording_id"].tolist()
    mapping = {rid: split for split, rids in result.items() for rid in rids}
    out = base.copy()
    out["split"] = out["recording_id"].map(mapping)
    if out["split"].isna().any():
        raise ValueError(f"Grabaciones sin asignar en {dataset} semilla {seed}")
    for a in SPLIT_NAMES:  # sin solapamiento de pacientes
        for b in SPLIT_NAMES:
            if a < b:
                shared = set(out.loc[out.split == a, "patient_group"]) & set(out.loc[out.split == b, "patient_group"])
                if shared:
                    raise ValueError(f"Pacientes compartidos {a}/{b} en {dataset} semilla {seed}: {sorted(shared)[:5]}")
    return out.sort_values(["split", "recording_id"]).reset_index(drop=True)


def original_spectrogram_index(dataset):
    root = ORIGINAL_SPEC_ROOT / f"{DATASETS[dataset]}_seed42_resize_dinamico_MEL64_SIZE64" / "npy"
    index = {}
    for split in SPLIT_NAMES:
        for path in (root / f"segmentado_{split}").glob("*/*.npy"):
            index[path.stem] = (path.parent.name, path)
    return index


def build_spectrogram_links(dataset, seed, split_df, index):
    dst = SPEC_DST / f"{DATASETS[dataset]}_seed{seed}_resize_dinamico_MEL64_SIZE64" / "npy"
    if dst.exists():
        raise FileExistsError(f"Ya existe {dst}; no se sobrescribe")
    split_of = dict(zip(split_df["recording_id"], split_df["split"]))
    counts = defaultdict(int)
    for stem, (label, src) in index.items():
        split = split_of[stem.rsplit("_ev", 1)[0]]
        target = dst / f"segmentado_{split}" / label
        target.mkdir(parents=True, exist_ok=True)
        os.symlink(src, target / src.name)
        counts[(split, label)] += 1
    return counts


def build_yamnet_embeddings(dataset, seed, split_df):
    src = EMB_SRC / f"r1_20261029_{dataset}_seed42_yamnet_looppad_mean_pool_1024"
    emb, lab, meta = [], [], []
    for split in SPLIT_NAMES:
        payload = np.load(src / f"segmentado_{split}_embeddings.npz")
        emb.append(payload["embeddings"]); lab.append(payload["labels"])
        meta.append(pd.read_csv(src / f"segmentado_{split}_metadata.csv"))
    emb, lab, meta = np.concatenate(emb), np.concatenate(lab), pd.concat(meta, ignore_index=True)
    recording = meta["segment_file"].str.replace(r"_ev\d+\.wav$", "", regex=True)
    new_split = recording.map(dict(zip(split_df["recording_id"], split_df["split"])))
    if new_split.isna().any():
        raise ValueError(f"Segmentos YAMNet sin particion en {dataset} semilla {seed}")
    out = EMB_DST / f"r1_{dataset}_seed{seed}_yamnet_looppad_mean_pool_1024"
    out.mkdir(parents=True, exist_ok=False)
    summary = {}
    for split in SPLIT_NAMES:
        mask = (new_split == split).to_numpy()
        np.savez_compressed(out / f"segmentado_{split}_embeddings.npz", embeddings=emb[mask], labels=lab[mask])
        meta.loc[mask].assign(split=split).to_csv(out / f"segmentado_{split}_metadata.csv", index=False)
        summary[split] = {"n": int(mask.sum()), "n_positive": int(lab[mask].sum())}
    with open(out / "embedding_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"source": str(src), "partition_seed": seed, "splits": summary}, f, indent=2)
    return summary


def main():
    report = []
    for dataset, name in DATASETS.items():
        base = pd.read_csv(SPLITS_DIR / name / "splits_patient_independent_seed42.csv")
        check = assign(dataset, base, 42)
        same = check.set_index("recording_id")["split"].sort_index().equals(base.set_index("recording_id")["split"].sort_index())
        print(f"{dataset}: semilla 42 reproduce el split original: {same}")
        if not same:
            raise SystemExit("La reconstruccion no reproduce el split original; no se generan particiones nuevas.")
        index = original_spectrogram_index(dataset)
        for seed in NEW_SEEDS:
            path = SPLITS_DIR / name / f"splits_patient_independent_seed{seed}.csv"
            if path.exists():
                raise FileExistsError(f"Ya existe {path}; no se sobrescribe")
            split_df = assign(dataset, base, seed)
            split_df.to_csv(path, index=False)
            build_spectrogram_links(dataset, seed, split_df, index)
            emb = build_yamnet_embeddings(dataset, seed, split_df)
            for split in SPLIT_NAMES:
                part = split_df[split_df.split == split]
                report.append({"dataset": dataset, "partition_seed": seed, "split": split,
                               "n_recordings": len(part), "n_patients": part["patient_group"].nunique(),
                               "n_events": emb[split]["n"], "n_positive": emb[split]["n_positive"],
                               "prevalence": emb[split]["n_positive"] / emb[split]["n"]})
        for split in SPLIT_NAMES:
            part = base[base.split == split]
            report.append({"dataset": dataset, "partition_seed": 42, "split": split, "n_recordings": len(part),
                           "n_patients": part["patient_group"].nunique(), "n_events": int(part["n_events"].sum()),
                           "n_positive": int(part["n_adventitious_events"].sum()),
                           "prevalence": part["n_adventitious_events"].sum() / part["n_events"].sum()})
    rep = pd.DataFrame(report).sort_values(["dataset", "partition_seed", "split"])
    Path("analisis/r1_20261029_fase3").mkdir(parents=True, exist_ok=True)
    rep.to_csv("analisis/r1_20261029_fase3/particiones_resumen.csv", index=False)
    print(rep.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
