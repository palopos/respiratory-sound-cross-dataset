"""R1 Fase 3 (R3.3, R4.CS2, opcion A1): wav2vec 2.0 afinado con el ajuste corregido, ICBHI -> SPRSound, 3 semillas.

Ajuste corregido (prueba piloto: codigo/train/wav2vec2/r1_Train_Wav2Vec2_Smoke.py): codificador convolucional congelado,
lr 1e-3 en la cabeza (projector + classifier) y 1e-5 en el transformer, pesos de clase balanceados.
Identico a r1_Train_Wav2Vec2_Fixed_SPRSound_to_ICBHI.py salvo los datos: train/validacion ICBHI (particion 42),
checkpoint y umbral solo con validacion de ICBHI; test oficial de SPRSound.
"""
import os
import sys

_SCRIPT_DIR = os.path.normpath(os.path.dirname(__file__) or ".")
sys.path.insert(0, _SCRIPT_DIR)

spr = "./ficheros/1 Segmentados_WAV/0 SPRSound/seed_42"
icbhi = "./ficheros/1 Segmentados_WAV/0 ICBHI/seed_42"
spr_csv = "./ficheros/splits/0 SPRSound/splits_patient_independent_seed42.csv"
icbhi_csv = "./ficheros/splits/0 ICBHI/splits_patient_independent_seed42.csv"
os.environ.update({
    "DATASET_NAME": "0 ICBHI", "DATASET_LABEL": "icbhi", "TRAIN_DATASET_LABEL": "icbhi",
    "VALIDATION_DATASET_LABEL": "icbhi", "TEST_DATASET_LABEL": "sprsound", "DATASET_PROTOCOL": "icbhi_to_sprsound",
    "EXPERIMENT_BLOCK": "r1_20261030_wav2vec2_finetune_fixed", "RUN_ID": "r1_20261030_fixed",
    "EXPERIMENT_NOTES": ("R1: ajuste corregido tras el colapso de 03_wav2vec2_finetune (encoder congelado, "
                         "lr cabeza 1e-3 / transformer 1e-5, pesos balanceados); 3 semillas."),
    "TRAIN_SEGMENTED_WAV_DIR": f"{icbhi}/segmentado_train", "VALIDATION_SEGMENTED_WAV_DIR": f"{icbhi}/segmentado_validation",
    "TEST_SEGMENTED_WAV_DIR": f"{spr}/segmentado_test",
    "EXPERIMENT_SEEDS": "42,43,44", "NUM_EPOCHS": "8", "EARLY_STOPPING_PATIENCE": "3",
    "LEARNING_RATE": "1e-5", "HEAD_LEARNING_RATE": "1e-3", "FREEZE_FEATURE_ENCODER": "1",
    "CLASS_WEIGHT_MODE": "balanced", "HF_LOCAL_FILES_ONLY": "1", "WAV2VEC2_DEVICE": "mps", "DISABLE_TQDM": "1",
})
for key in ("SPLIT_MANIFEST", "TRAIN_SPLIT_MANIFEST", "VALIDATION_SPLIT_MANIFEST"):
    os.environ[f"{key}_CSV"] = icbhi_csv
    os.environ[f"{key}_JSON"] = icbhi_csv.replace(".csv", ".json")
os.environ["TEST_SPLIT_MANIFEST_CSV"] = spr_csv
os.environ["TEST_SPLIT_MANIFEST_JSON"] = spr_csv.replace(".csv", ".json")

import Train_Wav2Vec2_Finetune as pipeline

if __name__ == "__main__":
    pipeline.main()
