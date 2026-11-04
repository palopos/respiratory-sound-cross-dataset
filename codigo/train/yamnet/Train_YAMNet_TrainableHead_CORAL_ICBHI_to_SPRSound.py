import os
import sys

_SCRIPT_DIR = os.path.normpath(os.path.dirname(__file__) or ".")
sys.path.insert(0, _SCRIPT_DIR)

_CORAL_EMBEDDINGS_DIR = "./ficheros/4 Domain Adaptation/0 YAMNet_CORAL_ICBHI_to_SPRSound_seed42"
_SPRSOUND_EMBEDDINGS_DIR = "./ficheros/3 Embeddings YAMNet/0 SPRSound_seed42_yamnet_tfhub_mean_pool_1024"
_ICBHI_SPLIT_JSON = "./ficheros/splits/0 ICBHI/splits_patient_independent_seed42.json"
_ICBHI_SPLIT_CSV = "./ficheros/splits/0 ICBHI/splits_patient_independent_seed42.csv"
_SPRSOUND_SPLIT_JSON = "./ficheros/splits/0 SPRSound/splits_patient_independent_seed42.json"
_SPRSOUND_SPLIT_CSV = "./ficheros/splits/0 SPRSound/splits_patient_independent_seed42.csv"

os.environ["DATASET_NAME"] = "0 ICBHI"
os.environ["DATASET_LABEL"] = "icbhi"
os.environ["TRAIN_DATASET_LABEL"] = "icbhi"
os.environ["VALIDATION_DATASET_LABEL"] = "icbhi"
os.environ["TEST_DATASET_LABEL"] = "sprsound"
os.environ["DATASET_PROTOCOL"] = "icbhi_to_sprsound"
os.environ["EXPERIMENT_BLOCK"] = "07_yamnet_coral_screening"
os.environ["TRAINING_MODE"] = "coral_frozen_backbone_trainable_head"
os.environ["EXPERIMENT_NOTES"] = (
    "Unsupervised domain adaptation screening with CORAL on frozen YAMNet embeddings. "
    "CORAL is fitted on ICBHI train embeddings and unlabeled SPRSound-train embeddings. "
    "Only source train/validation are transformed; test remains untouched. "
    "Checkpoint selection still uses source validation only."
)
os.environ["SPLIT_MANIFEST_JSON"] = _ICBHI_SPLIT_JSON
os.environ["SPLIT_MANIFEST_CSV"] = _ICBHI_SPLIT_CSV
os.environ["TRAIN_SPLIT_MANIFEST_JSON"] = _ICBHI_SPLIT_JSON
os.environ["TRAIN_SPLIT_MANIFEST_CSV"] = _ICBHI_SPLIT_CSV
os.environ["VALIDATION_SPLIT_MANIFEST_JSON"] = _ICBHI_SPLIT_JSON
os.environ["VALIDATION_SPLIT_MANIFEST_CSV"] = _ICBHI_SPLIT_CSV
os.environ["TEST_SPLIT_MANIFEST_JSON"] = _SPRSOUND_SPLIT_JSON
os.environ["TEST_SPLIT_MANIFEST_CSV"] = _SPRSOUND_SPLIT_CSV
os.environ["TRAIN_EMBEDDINGS_FILE"] = os.path.join(_CORAL_EMBEDDINGS_DIR, "segmentado_train_embeddings.npz")
os.environ["TRAIN_METADATA_FILE"] = os.path.join(_CORAL_EMBEDDINGS_DIR, "segmentado_train_metadata.csv")
os.environ["VALIDATION_EMBEDDINGS_FILE"] = os.path.join(_CORAL_EMBEDDINGS_DIR, "segmentado_validation_embeddings.npz")
os.environ["VALIDATION_METADATA_FILE"] = os.path.join(_CORAL_EMBEDDINGS_DIR, "segmentado_validation_metadata.csv")
os.environ["TEST_EMBEDDINGS_FILE"] = os.path.join(_SPRSOUND_EMBEDDINGS_DIR, "segmentado_test_embeddings.npz")
os.environ["TEST_METADATA_FILE"] = os.path.join(_SPRSOUND_EMBEDDINGS_DIR, "segmentado_test_metadata.csv")
os.environ["EXPERIMENT_SEEDS"] = "42"
os.environ["NUM_EPOCHS"] = "12"
os.environ["EARLY_STOPPING_PATIENCE"] = "4"

import Train_YAMNet_TrainableHead as pipeline


def main():
    pipeline.main()


if __name__ == "__main__":
    main()
