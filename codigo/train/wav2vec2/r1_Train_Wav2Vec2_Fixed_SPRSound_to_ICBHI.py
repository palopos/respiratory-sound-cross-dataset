"""R1 Fase 3 (R3.3, R4.CS2): wav2vec 2.0 afinado con el ajuste corregido, SPRSound -> ICBHI, 3 semillas.

Ajuste corregido (prueba piloto: codigo/train/wav2vec2/r1_Train_Wav2Vec2_Smoke.py): codificador convolucional congelado,
lr 1e-3 en la cabeza (projector + classifier) y 1e-5 en el transformer, pesos de clase balanceados.
Hasta 8 epocas, paciencia 3, checkpoint y umbral solo con validacion de SPRSound; particion 42; test ICBHI completo.
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
    "DATASET_NAME": "0 SPRSound", "DATASET_LABEL": "sprsound", "TRAIN_DATASET_LABEL": "sprsound",
    "VALIDATION_DATASET_LABEL": "sprsound", "TEST_DATASET_LABEL": "icbhi", "DATASET_PROTOCOL": "sprsound_to_icbhi",
    "EXPERIMENT_BLOCK": "r1_20261029_wav2vec2_finetune_fixed", "RUN_ID": "r1_20261029_fixed",
    "EXPERIMENT_NOTES": ("R1: ajuste corregido tras el colapso de 03_wav2vec2_finetune (encoder congelado, "
                         "lr cabeza 1e-3 / transformer 1e-5, pesos balanceados); 3 semillas."),
    "TRAIN_SEGMENTED_WAV_DIR": f"{spr}/segmentado_train", "VALIDATION_SEGMENTED_WAV_DIR": f"{spr}/segmentado_validation",
    "TEST_SEGMENTED_WAV_DIR": f"{icbhi}/segmentado_test",
    "EXPERIMENT_SEEDS": "42,43,44", "NUM_EPOCHS": "8", "EARLY_STOPPING_PATIENCE": "3",
    "LEARNING_RATE": "1e-5", "HEAD_LEARNING_RATE": "1e-3", "FREEZE_FEATURE_ENCODER": "1",
    "CLASS_WEIGHT_MODE": "balanced", "HF_LOCAL_FILES_ONLY": "1", "WAV2VEC2_DEVICE": "mps", "DISABLE_TQDM": "1",
})
for key in ("SPLIT_MANIFEST", "TRAIN_SPLIT_MANIFEST", "VALIDATION_SPLIT_MANIFEST"):
    os.environ[f"{key}_CSV"] = spr_csv
    os.environ[f"{key}_JSON"] = spr_csv.replace(".csv", ".json")
os.environ["TEST_SPLIT_MANIFEST_CSV"] = icbhi_csv
os.environ["TEST_SPLIT_MANIFEST_JSON"] = icbhi_csv.replace(".csv", ".json")

import Train_Wav2Vec2_Finetune as pipeline

if __name__ == "__main__":
    pipeline.main()
