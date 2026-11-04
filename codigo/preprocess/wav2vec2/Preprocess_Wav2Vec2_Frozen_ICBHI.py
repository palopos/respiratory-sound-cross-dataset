import os
import sys

_SCRIPT_DIR = os.path.normpath(os.path.dirname(__file__) or ".")
sys.path.insert(0, _SCRIPT_DIR)

os.environ.setdefault("DATASET_NAME", "0 ICBHI")
os.environ.setdefault("DATASET_LABEL", "icbhi")
os.environ.setdefault(
    "SEGMENTED_WAV_DIR",
    "./ficheros/1 Segmentados_WAV/0 ICBHI/seed_42",
)
os.environ.setdefault(
    "EMBEDDINGS_OUTPUT_DIR",
    "./ficheros/3 Embeddings wav2vec2/0 ICBHI_seed42_wav2vec2_hf_mean_pool_768",
)

import Preprocess_Wav2Vec2_Frozen as pipeline


def main():
    pipeline.main()


if __name__ == "__main__":
    main()
