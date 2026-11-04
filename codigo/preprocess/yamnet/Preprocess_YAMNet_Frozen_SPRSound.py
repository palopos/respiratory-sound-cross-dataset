import os
import sys

_SCRIPT_DIR = os.path.normpath(os.path.dirname(__file__) or ".")
sys.path.insert(0, _SCRIPT_DIR)

os.environ.setdefault("DATASET_NAME", "0 SPRSound")
os.environ.setdefault("DATASET_LABEL", "sprsound")
os.environ.setdefault(
    "SEGMENTED_WAV_DIR",
    "./ficheros/1 Segmentados_WAV/0 SPRSound/seed_42",
)
os.environ.setdefault(
    "EMBEDDINGS_OUTPUT_DIR",
    "./ficheros/3 Embeddings YAMNet/0 SPRSound_seed42_yamnet_tfhub_mean_pool_1024",
)

import Preprocess_YAMNet_Frozen as pipeline


def main():
    pipeline.main()


if __name__ == "__main__":
    main()
