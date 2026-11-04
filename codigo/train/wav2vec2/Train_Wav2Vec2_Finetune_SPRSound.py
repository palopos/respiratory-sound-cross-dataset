import os
import sys

_SCRIPT_DIR = os.path.normpath(os.path.dirname(__file__) or ".")
sys.path.insert(0, _SCRIPT_DIR)

_SPRSOUND_SEGMENTED_DIR = "./ficheros/1 Segmentados_WAV/0 SPRSound/seed_42"
_SPRSOUND_SPLIT_JSON = "./ficheros/splits/0 SPRSound/splits_patient_independent_seed42.json"
_SPRSOUND_SPLIT_CSV = "./ficheros/splits/0 SPRSound/splits_patient_independent_seed42.csv"

os.environ["DATASET_NAME"] = "0 SPRSound"
os.environ["DATASET_LABEL"] = "sprsound"
os.environ["TRAIN_DATASET_LABEL"] = "sprsound"
os.environ["VALIDATION_DATASET_LABEL"] = "sprsound"
os.environ["TEST_DATASET_LABEL"] = "sprsound"
os.environ["DATASET_PROTOCOL"] = "sprsound_to_sprsound"
os.environ["EXPERIMENT_NOTES"] = "Intra-dataset baseline: train, validation, and test on SPRSound."
os.environ["SPLIT_MANIFEST_JSON"] = _SPRSOUND_SPLIT_JSON
os.environ["SPLIT_MANIFEST_CSV"] = _SPRSOUND_SPLIT_CSV
os.environ["TRAIN_SEGMENTED_WAV_DIR"] = os.path.join(_SPRSOUND_SEGMENTED_DIR, "segmentado_train")
os.environ["VALIDATION_SEGMENTED_WAV_DIR"] = os.path.join(_SPRSOUND_SEGMENTED_DIR, "segmentado_validation")
os.environ["TEST_SEGMENTED_WAV_DIR"] = os.path.join(_SPRSOUND_SEGMENTED_DIR, "segmentado_test")

import Train_Wav2Vec2_Finetune as pipeline


def main():
    pipeline.main()


if __name__ == "__main__":
    main()
