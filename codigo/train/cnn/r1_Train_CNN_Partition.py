"""R1 Fase 3: CNN desde cero (3 semillas) con una particion alternativa; entrenamiento intra-dataset.

Uso: python r1_Train_CNN_Partition.py <icbhi|sprsound> <semilla_particion>
Espectrogramas: enlaces a los originales reorganizados por particion (codigo/fase3/r1_make_partitions.py).
Hiperparametros identicos a las ejecuciones originales (ICBHI pesos 1/1; SPRSound pesos 1/3.5, fijos en su script).
La evaluacion cruzada se hace despues con los checkpoints guardados (codigo/fase3/r1_CNN_Cross_Evaluate.py).
"""
import os
import sys

_SCRIPT_DIR = os.path.normpath(os.path.dirname(__file__) or ".")
sys.path.insert(0, _SCRIPT_DIR)

DATASETS = {"icbhi": "0 ICBHI", "sprsound": "0 SPRSound"}
source, split_seed = sys.argv[1], int(sys.argv[2])
if split_seed == 42:
    raise SystemExit("La particion 42 ya existe en las ejecuciones originales; no se repite.")
run_tag = os.environ.get("R1_RUN_TAG", "r1_20261029")
spec_dir = f"./ficheros/2 Espectrogramas/r1_particiones/{DATASETS[source]}_seed{split_seed}_resize_dinamico_MEL64_SIZE64"
split_csv = f"./ficheros/splits/{DATASETS[source]}/splits_patient_independent_seed{split_seed}.csv"

os.environ["DATASET_NAME"] = DATASETS[source]
os.environ["DATASET_LABEL"] = source
os.environ["TRAIN_DATASET_LABEL"] = source
os.environ["VALIDATION_DATASET_LABEL"] = source
os.environ["TEST_DATASET_LABEL"] = source
os.environ["DATASET_PROTOCOL"] = f"{source}_to_{source}"
os.environ["SPLIT_SEED"] = str(split_seed)
os.environ["EXPERIMENT_BLOCK"] = f"{run_tag}_cnn_partitions"
os.environ["RUN_ID"] = f"{run_tag}_split{split_seed}"
os.environ["EXPERIMENT_NOTES"] = f"R1 Fase 3: particion alternativa por paciente (semilla {split_seed}); misma configuracion que 00_cnn_scratch_baseline."
for key in ("SPLIT_MANIFEST", "TRAIN_SPLIT_MANIFEST", "VALIDATION_SPLIT_MANIFEST", "TEST_SPLIT_MANIFEST"):
    os.environ[f"{key}_CSV"] = split_csv
    os.environ[f"{key}_JSON"] = split_csv.replace(".csv", ".json")
os.environ["PATH_SALIDA_NPY"] = spec_dir
for split in ("train", "validation", "test"):
    os.environ[f"{split.upper()}_NPY_PATH"] = os.path.join(spec_dir, "npy", f"segmentado_{split}")
os.environ["EXPERIMENT_SEEDS"] = "42,43,44"
if source == "icbhi":
    os.environ["CLASS_WEIGHT_NEGATIVE"] = "1.0"
    os.environ["CLASS_WEIGHT_POSITIVE"] = "1.0"
    os.environ["EPOCHS"] = "30"
    import Train_CNN_Scratch_ICBHI as pipeline
else:
    import Train_CNN_Scratch_SPRSound as pipeline

if __name__ == "__main__":
    pipeline.main()
