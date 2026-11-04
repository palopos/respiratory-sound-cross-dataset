import os
import sys

_SCRIPT_DIR = os.path.normpath(os.path.dirname(__file__) or ".")
sys.path.insert(0, _SCRIPT_DIR)

_FUSION_EMBEDDINGS_DIR = "./ficheros/3 Embeddings YAMNet/0 Fusion SPRSound ICBHI_seed42_yamnet_tfhub_mean_pool_1024"
_FUSION_SPLIT_JSON = "./ficheros/splits/0 Fusion SPRSound ICBHI/splits_patient_independent_seed42.json"
_FUSION_SPLIT_CSV = "./ficheros/splits/0 Fusion SPRSound ICBHI/splits_patient_independent_seed42.csv"

os.environ.setdefault("DATASET_NAME", "0 Fusion SPRSound ICBHI")
os.environ.setdefault("DATASET_LABEL", "sprsound_icbhi_fusion")
os.environ.setdefault("TRAIN_DATASET_LABEL", "sprsound_icbhi_fusion")
os.environ.setdefault("VALIDATION_DATASET_LABEL", "sprsound_icbhi_fusion")
os.environ.setdefault("TEST_DATASET_LABEL", "sprsound_icbhi_fusion")
os.environ.setdefault("DATASET_PROTOCOL", "sprsound_icbhi_fusion_to_sprsound_icbhi_fusion")
os.environ.setdefault("EMBEDDINGS_BASE_DIR", _FUSION_EMBEDDINGS_DIR)
os.environ.setdefault("SPLIT_MANIFEST_JSON", _FUSION_SPLIT_JSON)
os.environ.setdefault("SPLIT_MANIFEST_CSV", _FUSION_SPLIT_CSV)

import Train_YAMNet_Frozen as pipeline


def main():
    pipeline.main()


if __name__ == "__main__":
    main()
