import os
import sys

_SCRIPT_DIR = os.path.normpath(os.path.dirname(__file__) or ".")
sys.path.insert(0, _SCRIPT_DIR)

_ICBHI_SPECTROGRAM_DIR = "./ficheros/2 Espectrogramas/0 ICBHI_seed42_resize_dinamico_MEL64_SIZE64"
_SPRSOUND_SPECTROGRAM_DIR = "./ficheros/2 Espectrogramas/0 SPRSound_seed42_resize_dinamico_MEL64_SIZE64"
_ICBHI_SPLIT_JSON = "./ficheros/splits/0 ICBHI/splits_patient_independent_seed42.json"
_ICBHI_SPLIT_CSV = "./ficheros/splits/0 ICBHI/splits_patient_independent_seed42.csv"
_SPRSOUND_SPLIT_JSON = "./ficheros/splits/0 SPRSound/splits_patient_independent_seed42.json"
_SPRSOUND_SPLIT_CSV = "./ficheros/splits/0 SPRSound/splits_patient_independent_seed42.csv"

os.environ.setdefault("DATASET_PROTOCOL", "icbhi_to_sprsound")
os.environ.setdefault("DATASET_LABEL", "icbhi")
os.environ.setdefault("TRAIN_DATASET_LABEL", "icbhi")
os.environ.setdefault("VALIDATION_DATASET_LABEL", "icbhi")
os.environ.setdefault("TEST_DATASET_LABEL", "sprsound")
os.environ.setdefault(
    "EXPERIMENT_NOTES",
    "Cruce puro: train y validation desde ICBHI; test externo en SPRSound.",
)
os.environ.setdefault("SPLIT_MANIFEST_JSON", _ICBHI_SPLIT_JSON)
os.environ.setdefault("SPLIT_MANIFEST_CSV", _ICBHI_SPLIT_CSV)
os.environ.setdefault("TRAIN_SPLIT_MANIFEST_JSON", _ICBHI_SPLIT_JSON)
os.environ.setdefault("TRAIN_SPLIT_MANIFEST_CSV", _ICBHI_SPLIT_CSV)
os.environ.setdefault("VALIDATION_SPLIT_MANIFEST_JSON", _ICBHI_SPLIT_JSON)
os.environ.setdefault("VALIDATION_SPLIT_MANIFEST_CSV", _ICBHI_SPLIT_CSV)
os.environ.setdefault("TEST_SPLIT_MANIFEST_JSON", _SPRSOUND_SPLIT_JSON)
os.environ.setdefault("TEST_SPLIT_MANIFEST_CSV", _SPRSOUND_SPLIT_CSV)
os.environ.setdefault("PATH_SALIDA_NPY", _ICBHI_SPECTROGRAM_DIR)
os.environ.setdefault("TRAIN_NPY_PATH", os.path.join(_ICBHI_SPECTROGRAM_DIR, "npy", "segmentado_train"))
os.environ.setdefault("VALIDATION_NPY_PATH", os.path.join(_ICBHI_SPECTROGRAM_DIR, "npy", "segmentado_validation"))
os.environ.setdefault("TEST_NPY_PATH", os.path.join(_SPRSOUND_SPECTROGRAM_DIR, "npy", "segmentado_test"))

import Train_CNN_Scratch_ICBHI as pipeline


def main():
    pipeline.main()


if __name__ == "__main__":
    main()
