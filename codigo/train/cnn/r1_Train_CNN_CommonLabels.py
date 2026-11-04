"""R1 Fase 3 (R3.4): CNN desde cero (3 semillas) entrenada en SPRSound sin Rhonchi/Stridor, particion 42.

Uso: python r1_Train_CNN_CommonLabels.py   (la evaluacion en ICBHI se hace despues con r1_CNN_Cross_Evaluate.py)
"""
import os
import sys

_SCRIPT_DIR = os.path.normpath(os.path.dirname(__file__) or ".")
sys.path.insert(0, _SCRIPT_DIR)

run_tag = os.environ.get("R1_RUN_TAG", "r1_20261029")
spec_dir = "./ficheros/2 Espectrogramas/r1_comunes/0 SPRSound_seed42_commonlabels_resize_dinamico_MEL64_SIZE64"
split_csv = "./ficheros/splits/0 SPRSound/splits_patient_independent_seed42.csv"
os.environ["DATASET_NAME"] = "0 SPRSound"
os.environ["DATASET_LABEL"] = "sprsound"
os.environ["DATASET_PROTOCOL"] = "sprsound_to_sprsound"
os.environ["SPLIT_SEED"] = "42"
os.environ["EXPERIMENT_BLOCK"] = f"{run_tag}_cnn_common_labels"
os.environ["RUN_ID"] = f"{run_tag}_commonlabels"
os.environ["EXPERIMENT_NOTES"] = "R1 Fase 3: SPRSound sin Rhonchi/Stridor (etiquetas compartidas con ICBHI); configuracion de 00_cnn_scratch_baseline."
for key in ("SPLIT_MANIFEST", "TRAIN_SPLIT_MANIFEST", "VALIDATION_SPLIT_MANIFEST", "TEST_SPLIT_MANIFEST"):
    os.environ[f"{key}_CSV"] = split_csv
    os.environ[f"{key}_JSON"] = split_csv.replace(".csv", ".json")
os.environ["PATH_SALIDA_NPY"] = spec_dir
for split in ("train", "validation", "test"):
    os.environ[f"{split.upper()}_NPY_PATH"] = os.path.join(spec_dir, "npy", f"segmentado_{split}")
os.environ["EXPERIMENT_SEEDS"] = "42,43,44"

import Train_CNN_Scratch_SPRSound as pipeline

if __name__ == "__main__":
    pipeline.main()
