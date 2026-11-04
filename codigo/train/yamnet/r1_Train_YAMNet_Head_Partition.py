"""R1 Fase 3: YAMNet trainable head (relleno loop, 3 semillas) con particiones alternativas.

Uso: python r1_Train_YAMNet_Head_Partition.py <protocolo> <semilla_particion>
  protocolo: icbhi_to_icbhi | icbhi_to_sprsound | sprsound_to_sprsound | sprsound_to_icbhi
En los protocolos cruzados el test es el de la misma semilla de particion en el dataset destino
(en SPRSound el test oficial no cambia). Configuracion identica a la Fase 1.
"""
import os
import sys

_SCRIPT_DIR = os.path.normpath(os.path.dirname(__file__) or ".")
sys.path.insert(0, _SCRIPT_DIR)

DATASETS = {"icbhi": "0 ICBHI", "sprsound": "0 SPRSound"}
protocol, split_seed = sys.argv[1], int(sys.argv[2])
train_label, test_label = protocol.split("_to_")
run_tag = os.environ.get("R1_RUN_TAG", "r1_20261029")


def embeddings_dir(dataset_label):
    return f"./ficheros/3 Embeddings YAMNet/r1_particiones/r1_{dataset_label}_seed{split_seed}_yamnet_looppad_mean_pool_1024"


def split_files(dataset_label):
    base = f"./ficheros/splits/{DATASETS[dataset_label]}/splits_patient_independent_seed{split_seed}"
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
os.environ["SPLIT_SEED"] = str(split_seed)
os.environ["EXPERIMENT_BLOCK"] = f"{run_tag}_yamnet_head_partitions"
os.environ["RUN_ID"] = f"{run_tag}_loop_split{split_seed}"
os.environ["EMBEDDING_MODE"] = "yamnet_mean_pool_1024_looppad"
os.environ["EXPERIMENT_NOTES"] = (
    f"R1 Fase 3: particion alternativa (semilla {split_seed}), relleno loop. "
    "Misma configuracion que 04_yamnet_trainable_head_confirm (semillas 42-44, 12 epocas, paciencia 4)."
)
for key, value in (("SPLIT_MANIFEST", (train_json, train_csv)), ("TRAIN_SPLIT_MANIFEST", (train_json, train_csv)),
                   ("VALIDATION_SPLIT_MANIFEST", (train_json, train_csv)), ("TEST_SPLIT_MANIFEST", (test_json, test_csv))):
    os.environ[f"{key}_JSON"], os.environ[f"{key}_CSV"] = value
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
