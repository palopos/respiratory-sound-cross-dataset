"""R1 Fase 1: YAMNet trainable head (3 semillas) con embeddings de distintos rellenos.

Uso: python r1_Train_YAMNet_Head_Padding.py <protocolo> <variante>
  protocolo: icbhi_to_icbhi | icbhi_to_sprsound | sprsound_to_sprsound | sprsound_to_icbhi
  variante:  zero (embeddings originales) | loop | noise | min0975 (solo eventos >=0.975 s)
             | loopcommon (loop; SPRSound sin Rhonchi/Stridor)
Configuracion identica a 04_yamnet_trainable_head_confirm: semillas 42-44, 12 epocas, paciencia 4.
"""
import os
import sys

_SCRIPT_DIR = os.path.normpath(os.path.dirname(__file__) or ".")
sys.path.insert(0, _SCRIPT_DIR)

DATASETS = {"icbhi": "0 ICBHI", "sprsound": "0 SPRSound"}
protocol, variant = sys.argv[1], sys.argv[2]
train_label, test_label = protocol.split("_to_")
run_tag = os.environ.get("R1_RUN_TAG", "r1_20261029")


def embeddings_dir(dataset_label):
    if variant == "zero":
        return f"./ficheros/3 Embeddings YAMNet/{DATASETS[dataset_label]}_seed42_yamnet_tfhub_mean_pool_1024"
    if variant in ("loop", "noise"):
        return f"./ficheros/3 Embeddings YAMNet/{run_tag}_{dataset_label}_seed42_yamnet_{variant}pad_mean_pool_1024"
    if variant == "loopcommon":  # Fase 3: SPRSound sin Rhonchi/Stridor (codigo/fase3/r1_make_common_labels.py)
        if dataset_label == "sprsound":
            return f"./ficheros/3 Embeddings YAMNet/{run_tag}_sprsound_seed42_yamnet_looppad_commonlabels"
        return f"./ficheros/3 Embeddings YAMNet/{run_tag}_{dataset_label}_seed42_yamnet_looppad_mean_pool_1024"
    if variant == "min0975":
        return f"./ficheros/3 Embeddings YAMNet/{run_tag}_{dataset_label}_seed42_yamnet_min0975_mean_pool_1024"
    raise ValueError(f"Variante no soportada: {variant}")


def split_files(dataset_label):
    base = f"./ficheros/splits/{DATASETS[dataset_label]}/splits_patient_independent_seed42"
    return f"{base}.json", f"{base}.csv"


train_dir, test_dir = embeddings_dir(train_label), embeddings_dir(test_label)
train_json, train_csv = split_files(train_label)
test_json, test_csv = split_files(test_label)

os.environ["DATASET_NAME"] = DATASETS[train_label]
os.environ["DATASET_LABEL"] = train_label
os.environ["TRAIN_DATASET_LABEL"] = train_label
os.environ["VALIDATION_DATASET_LABEL"] = train_label
os.environ["TEST_DATASET_LABEL"] = test_label
os.environ["DATASET_PROTOCOL"] = protocol
os.environ["EXPERIMENT_BLOCK"] = f"{run_tag}_yamnet_head_padding_{variant}"
os.environ["RUN_ID"] = f"{run_tag}_{variant}"
os.environ["EMBEDDING_MODE"] = f"yamnet_mean_pool_1024_{variant}pad"
os.environ["EXPERIMENT_NOTES"] = (
    f"R1 Fase 1: YAMNet trainable head con relleno '{variant}'. "
    "Misma configuracion que 04_yamnet_trainable_head_confirm (semillas 42-44, 12 epocas, paciencia 4)."
)
os.environ["SPLIT_MANIFEST_JSON"] = train_json
os.environ["SPLIT_MANIFEST_CSV"] = train_csv
os.environ["TRAIN_SPLIT_MANIFEST_JSON"] = train_json
os.environ["TRAIN_SPLIT_MANIFEST_CSV"] = train_csv
os.environ["VALIDATION_SPLIT_MANIFEST_JSON"] = train_json
os.environ["VALIDATION_SPLIT_MANIFEST_CSV"] = train_csv
os.environ["TEST_SPLIT_MANIFEST_JSON"] = test_json
os.environ["TEST_SPLIT_MANIFEST_CSV"] = test_csv
for split, directory in (("TRAIN", train_dir), ("VALIDATION", train_dir), ("TEST", test_dir)):
    prefix = f"segmentado_{split.lower()}"
    os.environ[f"{split}_EMBEDDINGS_FILE"] = os.path.join(directory, f"{prefix}_embeddings.npz")
    os.environ[f"{split}_METADATA_FILE"] = os.path.join(directory, f"{prefix}_metadata.csv")
os.environ["EXPERIMENT_SEEDS"] = "42,43,44"
os.environ["NUM_EPOCHS"] = "12"
os.environ["EARLY_STOPPING_PATIENCE"] = "4"

import Train_YAMNet_TrainableHead as pipeline

if __name__ == "__main__":
    pipeline.main()
