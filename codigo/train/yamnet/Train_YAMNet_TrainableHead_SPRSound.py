import os
import sys

_SCRIPT_DIR = os.path.normpath(os.path.dirname(__file__) or ".")
sys.path.insert(0, _SCRIPT_DIR)

_SPRSOUND_EMBEDDINGS_DIR = "./ficheros/3 Embeddings YAMNet/0 SPRSound_seed42_yamnet_tfhub_mean_pool_1024"
_SPRSOUND_SPLIT_JSON = "./ficheros/splits/0 SPRSound/splits_patient_independent_seed42.json"
_SPRSOUND_SPLIT_CSV = "./ficheros/splits/0 SPRSound/splits_patient_independent_seed42.csv"

os.environ["DATASET_NAME"] = "0 SPRSound"
os.environ["DATASET_LABEL"] = "sprsound"
os.environ["TRAIN_DATASET_LABEL"] = "sprsound"
os.environ["VALIDATION_DATASET_LABEL"] = "sprsound"
os.environ["TEST_DATASET_LABEL"] = "sprsound"
os.environ["DATASET_PROTOCOL"] = "sprsound_to_sprsound"
os.environ["EXPERIMENT_NOTES"] = "Intra-dataset screening with frozen YAMNet embeddings and a trainable head on SPRSound."
os.environ["SPLIT_MANIFEST_JSON"] = _SPRSOUND_SPLIT_JSON
os.environ["SPLIT_MANIFEST_CSV"] = _SPRSOUND_SPLIT_CSV
os.environ["TRAIN_SPLIT_MANIFEST_JSON"] = _SPRSOUND_SPLIT_JSON
os.environ["TRAIN_SPLIT_MANIFEST_CSV"] = _SPRSOUND_SPLIT_CSV
os.environ["VALIDATION_SPLIT_MANIFEST_JSON"] = _SPRSOUND_SPLIT_JSON
os.environ["VALIDATION_SPLIT_MANIFEST_CSV"] = _SPRSOUND_SPLIT_CSV
os.environ["TEST_SPLIT_MANIFEST_JSON"] = _SPRSOUND_SPLIT_JSON
os.environ["TEST_SPLIT_MANIFEST_CSV"] = _SPRSOUND_SPLIT_CSV
os.environ["TRAIN_EMBEDDINGS_FILE"] = os.path.join(_SPRSOUND_EMBEDDINGS_DIR, "segmentado_train_embeddings.npz")
os.environ["TRAIN_METADATA_FILE"] = os.path.join(_SPRSOUND_EMBEDDINGS_DIR, "segmentado_train_metadata.csv")
os.environ["VALIDATION_EMBEDDINGS_FILE"] = os.path.join(_SPRSOUND_EMBEDDINGS_DIR, "segmentado_validation_embeddings.npz")
os.environ["VALIDATION_METADATA_FILE"] = os.path.join(_SPRSOUND_EMBEDDINGS_DIR, "segmentado_validation_metadata.csv")
os.environ["TEST_EMBEDDINGS_FILE"] = os.path.join(_SPRSOUND_EMBEDDINGS_DIR, "segmentado_test_embeddings.npz")
os.environ["TEST_METADATA_FILE"] = os.path.join(_SPRSOUND_EMBEDDINGS_DIR, "segmentado_test_metadata.csv")

import Train_YAMNet_TrainableHead as pipeline


def main():
    pipeline.main()


if __name__ == "__main__":
    main()
