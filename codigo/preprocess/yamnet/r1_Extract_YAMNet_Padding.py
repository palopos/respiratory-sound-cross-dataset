"""R1 Fase 1: reextrae embeddings YAMNet con distintos rellenos para eventos <0.975 s.

Uso (desde cualquier sitio): python r1_Extract_YAMNet_Padding.py <icbhi|sprsound> <zero|loop|noise>
La salida va a ficheros/3 Embeddings YAMNet/r1_20261029_<dataset>_seed42_yamnet_<pad>pad_mean_pool_1024.
"""
import os
import sys

_SCRIPT_DIR = os.path.normpath(os.path.dirname(__file__) or ".")
sys.path.insert(0, _SCRIPT_DIR)

DATASETS = {"icbhi": "0 ICBHI", "sprsound": "0 SPRSound"}
dataset_label, pad_mode = sys.argv[1], sys.argv[2]
dataset_name = DATASETS[dataset_label]
run_tag = os.environ.get("R1_RUN_TAG", "r1_20261029")

os.environ["DATASET_NAME"] = dataset_name
os.environ["DATASET_LABEL"] = dataset_label
os.environ["PAD_MODE"] = pad_mode
os.environ["SEGMENTED_WAV_DIR"] = f"./ficheros/1 Segmentados_WAV/{dataset_name}/seed_42"
os.environ["EMBEDDINGS_OUTPUT_DIR"] = (
    f"./ficheros/3 Embeddings YAMNet/{run_tag}_{dataset_label}_seed42_yamnet_{pad_mode}pad_mean_pool_1024"
)
os.environ["EXPERIMENT_NOTES"] = f"R1 Fase 1: reextraccion YAMNet con PAD_MODE={pad_mode} (eventos <0.975 s)."

import Preprocess_YAMNet_Frozen as pipeline

if __name__ == "__main__":
    pipeline.main()
