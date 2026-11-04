import os
import sys

_SCRIPT_DIR = os.path.normpath(os.path.dirname(__file__) or ".")
sys.path.insert(0, _SCRIPT_DIR)

os.environ["SOURCE_DATASET_LABEL"] = "icbhi"
os.environ["TARGET_DATASET_LABEL"] = "sprsound"
os.environ["SOURCE_EMBEDDINGS_DIR"] = "./ficheros/3 Embeddings YAMNet/0 ICBHI_seed42_yamnet_tfhub_mean_pool_1024"
os.environ["TARGET_EMBEDDINGS_DIR"] = "./ficheros/3 Embeddings YAMNet/0 SPRSound_seed42_yamnet_tfhub_mean_pool_1024"
os.environ["OUTPUT_DIR"] = "./ficheros/4 Domain Adaptation/0 YAMNet_CORAL_ICBHI_to_SPRSound_seed42"
os.environ["TARGET_REFERENCE_SPLITS"] = "train"

import Prepare_YAMNet_CORAL_Embeddings as pipeline


def main():
    pipeline.main()


if __name__ == "__main__":
    main()
