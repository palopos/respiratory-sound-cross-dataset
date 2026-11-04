"""R1 Fase 3: prueba corta del ajuste corregido de wav2vec 2.0 (SPRSound, particion 42, 1 semilla).

Ajuste: codificador convolucional congelado, lr 1e-3 en la cabeza y 1e-5 en el transformer, pesos de clase
balanceados. Prueba: 1500 eventos de train estratificados, 2 epocas, validacion completa; test reducido a 50
eventos (no se usa para decidir nada).
Criterio fijado antes de ejecutar: perdida de train claramente < ln 2 (0.693), DE de las probabilidades de
validacion > 0.05 y F1 de validacion > 0.
"""
import os
import sys

_SCRIPT_DIR = os.path.normpath(os.path.dirname(__file__) or ".")
sys.path.insert(0, _SCRIPT_DIR)

seg = "./ficheros/1 Segmentados_WAV/0 SPRSound/seed_42"
split_csv = "./ficheros/splits/0 SPRSound/splits_patient_independent_seed42.csv"
os.environ.update({
    "DATASET_NAME": "0 SPRSound", "DATASET_LABEL": "sprsound", "TRAIN_DATASET_LABEL": "sprsound",
    "VALIDATION_DATASET_LABEL": "sprsound", "TEST_DATASET_LABEL": "sprsound", "DATASET_PROTOCOL": "sprsound_to_sprsound",
    "EXPERIMENT_BLOCK": "r1_20261029_wav2vec2_finetune_smoke", "RUN_ID": "r1_20261029_smoke",
    "EXPERIMENT_NOTES": "R1 prueba corta: encoder congelado, lr cabeza 1e-3 / transformer 1e-5, pesos balanceados.",
    "TRAIN_SEGMENTED_WAV_DIR": f"{seg}/segmentado_train", "VALIDATION_SEGMENTED_WAV_DIR": f"{seg}/segmentado_validation",
    "TEST_SEGMENTED_WAV_DIR": f"{seg}/segmentado_test",
    "EXPERIMENT_SEEDS": "42", "NUM_EPOCHS": "2", "EARLY_STOPPING_PATIENCE": "2",
    "LEARNING_RATE": "1e-5", "HEAD_LEARNING_RATE": "1e-3", "FREEZE_FEATURE_ENCODER": "1",
    "CLASS_WEIGHT_MODE": "balanced", "MAX_TRAIN_SAMPLES": "1500", "MAX_TEST_SAMPLES": "50",
    "HF_LOCAL_FILES_ONLY": "1", "WAV2VEC2_DEVICE": "mps", "DISABLE_TQDM": "1",
})
for key in ("SPLIT_MANIFEST", "TRAIN_SPLIT_MANIFEST", "VALIDATION_SPLIT_MANIFEST", "TEST_SPLIT_MANIFEST"):
    os.environ[f"{key}_CSV"] = split_csv
    os.environ[f"{key}_JSON"] = split_csv.replace(".csv", ".json")

import Train_Wav2Vec2_Finetune as pipeline

if __name__ == "__main__":
    pipeline.main()
