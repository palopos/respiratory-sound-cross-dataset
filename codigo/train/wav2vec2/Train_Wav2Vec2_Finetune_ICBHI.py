import os
import sys

_SCRIPT_DIR = os.path.normpath(os.path.dirname(__file__) or ".")
sys.path.insert(0, _SCRIPT_DIR)

_ICBHI_SEGMENTED_DIR = "./ficheros/1 Segmentados_WAV/0 ICBHI/seed_42"
_ICBHI_SPLIT_JSON = "./ficheros/splits/0 ICBHI/splits_patient_independent_seed42.json"
_ICBHI_SPLIT_CSV = "./ficheros/splits/0 ICBHI/splits_patient_independent_seed42.csv"

os.environ["DATASET_NAME"] = "0 ICBHI"
os.environ["DATASET_LABEL"] = "icbhi"
os.environ["TRAIN_DATASET_LABEL"] = "icbhi"
os.environ["VALIDATION_DATASET_LABEL"] = "icbhi"
os.environ["TEST_DATASET_LABEL"] = "icbhi"
os.environ["DATASET_PROTOCOL"] = "icbhi_to_icbhi"
os.environ["EXPERIMENT_NOTES"] = "Intra-dataset baseline: train, validation, and test on ICBHI."
os.environ["SPLIT_MANIFEST_JSON"] = _ICBHI_SPLIT_JSON
os.environ["SPLIT_MANIFEST_CSV"] = _ICBHI_SPLIT_CSV
os.environ["TRAIN_SEGMENTED_WAV_DIR"] = os.path.join(_ICBHI_SEGMENTED_DIR, "segmentado_train")
os.environ["VALIDATION_SEGMENTED_WAV_DIR"] = os.path.join(_ICBHI_SEGMENTED_DIR, "segmentado_validation")
os.environ["TEST_SEGMENTED_WAV_DIR"] = os.path.join(_ICBHI_SEGMENTED_DIR, "segmentado_test")

import Train_Wav2Vec2_Finetune as pipeline


def main():
    pipeline.main()


if __name__ == "__main__":
    main()
