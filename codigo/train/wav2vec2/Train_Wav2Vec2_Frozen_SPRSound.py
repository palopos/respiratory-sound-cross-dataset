import os
import sys

_SCRIPT_DIR = os.path.normpath(os.path.dirname(__file__) or ".")
sys.path.insert(0, _SCRIPT_DIR)

_SPRSOUND_EMBEDDINGS_DIR = "./ficheros/3 Embeddings wav2vec2/0 SPRSound_seed42_wav2vec2_hf_mean_pool_768"
_SPRSOUND_SPLIT_JSON = "./ficheros/splits/0 SPRSound/splits_patient_independent_seed42.json"
_SPRSOUND_SPLIT_CSV = "./ficheros/splits/0 SPRSound/splits_patient_independent_seed42.csv"

os.environ.setdefault("DATASET_NAME", "0 SPRSound")
os.environ.setdefault("DATASET_LABEL", "sprsound")
os.environ.setdefault("TRAIN_DATASET_LABEL", "sprsound")
os.environ.setdefault("VALIDATION_DATASET_LABEL", "sprsound")
os.environ.setdefault("TEST_DATASET_LABEL", "sprsound")
os.environ.setdefault("DATASET_PROTOCOL", "sprsound_to_sprsound")
os.environ.setdefault("EMBEDDINGS_BASE_DIR", _SPRSOUND_EMBEDDINGS_DIR)
os.environ.setdefault("SPLIT_MANIFEST_JSON", _SPRSOUND_SPLIT_JSON)
os.environ.setdefault("SPLIT_MANIFEST_CSV", _SPRSOUND_SPLIT_CSV)

import Train_Wav2Vec2_Frozen as pipeline


def main():
    pipeline.main()


if __name__ == "__main__":
    main()
