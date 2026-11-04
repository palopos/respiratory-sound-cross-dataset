import os
import sys

_SCRIPT_DIR = os.path.normpath(os.path.dirname(__file__) or ".")
sys.path.insert(0, _SCRIPT_DIR)

_FUSION_EMBEDDINGS_DIR = "./ficheros/3 Embeddings wav2vec2/0 Fusion SPRSound ICBHI_seed42_wav2vec2_hf_mean_pool_768"
_ICBHI_EMBEDDINGS_DIR = "./ficheros/3 Embeddings wav2vec2/0 ICBHI_seed42_wav2vec2_hf_mean_pool_768"
_FUSION_SPLIT_JSON = "./ficheros/splits/0 Fusion SPRSound ICBHI/splits_patient_independent_seed42.json"
_FUSION_SPLIT_CSV = "./ficheros/splits/0 Fusion SPRSound ICBHI/splits_patient_independent_seed42.csv"
_ICBHI_SPLIT_JSON = "./ficheros/splits/0 ICBHI/splits_patient_independent_seed42.json"
_ICBHI_SPLIT_CSV = "./ficheros/splits/0 ICBHI/splits_patient_independent_seed42.csv"

os.environ.setdefault("DATASET_NAME", "0 Fusion SPRSound ICBHI")
os.environ.setdefault("DATASET_LABEL", "sprsound_icbhi_fusion")
os.environ.setdefault("TRAIN_DATASET_LABEL", "sprsound_icbhi_fusion")
os.environ.setdefault("VALIDATION_DATASET_LABEL", "icbhi")
os.environ.setdefault("TEST_DATASET_LABEL", "icbhi")
os.environ.setdefault("DATASET_PROTOCOL", "sprsound_icbhi_fusion_target_tuned_to_icbhi")
os.environ.setdefault("EXPERIMENT_NOTES", "Fusion-target-tuned: train desde Fusion; validation y test desde ICBHI.")
os.environ.setdefault("EMBEDDINGS_BASE_DIR", _FUSION_EMBEDDINGS_DIR)
os.environ.setdefault("SPLIT_MANIFEST_JSON", _FUSION_SPLIT_JSON)
os.environ.setdefault("SPLIT_MANIFEST_CSV", _FUSION_SPLIT_CSV)
os.environ.setdefault("TRAIN_SPLIT_MANIFEST_JSON", _FUSION_SPLIT_JSON)
os.environ.setdefault("TRAIN_SPLIT_MANIFEST_CSV", _FUSION_SPLIT_CSV)
os.environ.setdefault("VALIDATION_SPLIT_MANIFEST_JSON", _ICBHI_SPLIT_JSON)
os.environ.setdefault("VALIDATION_SPLIT_MANIFEST_CSV", _ICBHI_SPLIT_CSV)
os.environ.setdefault("TEST_SPLIT_MANIFEST_JSON", _ICBHI_SPLIT_JSON)
os.environ.setdefault("TEST_SPLIT_MANIFEST_CSV", _ICBHI_SPLIT_CSV)
os.environ.setdefault("TRAIN_EMBEDDINGS_FILE", os.path.join(_FUSION_EMBEDDINGS_DIR, "segmentado_train_embeddings.npz"))
os.environ.setdefault("TRAIN_METADATA_FILE", os.path.join(_FUSION_EMBEDDINGS_DIR, "segmentado_train_metadata.csv"))
os.environ.setdefault("VALIDATION_EMBEDDINGS_FILE", os.path.join(_ICBHI_EMBEDDINGS_DIR, "segmentado_validation_embeddings.npz"))
os.environ.setdefault("VALIDATION_METADATA_FILE", os.path.join(_ICBHI_EMBEDDINGS_DIR, "segmentado_validation_metadata.csv"))
os.environ.setdefault("TEST_EMBEDDINGS_FILE", os.path.join(_ICBHI_EMBEDDINGS_DIR, "segmentado_test_embeddings.npz"))
os.environ.setdefault("TEST_METADATA_FILE", os.path.join(_ICBHI_EMBEDDINGS_DIR, "segmentado_test_metadata.csv"))

import Train_Wav2Vec2_Frozen as pipeline


def main():
    pipeline.main()


if __name__ == "__main__":
    main()
