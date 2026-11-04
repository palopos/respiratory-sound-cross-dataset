from __future__ import annotations

from datetime import datetime
import copy
import glob
import json
import math
import os
import random
import re
from functools import lru_cache

# Localiza la raiz del proyecto para mantener rutas relativas a Fix_Paper1.
_SCRIPT_DIR = os.path.normpath(os.path.dirname(__file__) or ".")


def _find_project_root(start_dir):
    current_dir = os.path.abspath(start_dir)
    while True:
        if os.path.exists(os.path.join(current_dir, "environment.yml")) and os.path.isdir(os.path.join(current_dir, "ficheros")):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise FileNotFoundError("No se pudo localizar la raiz de Fix_Paper1.")
        current_dir = parent_dir


_PROJECT_ROOT = _find_project_root(_SCRIPT_DIR)
os.chdir(_PROJECT_ROOT)
_PROJECT_DIR = "."
_CACHE_DIR = os.path.join(_PROJECT_DIR, ".cache")
_HF_CACHE_DIR = os.path.join(_CACHE_DIR, "huggingface")
_MPLCONFIGDIR = os.path.join(_PROJECT_DIR, ".matplotlib")
_DEFAULT_CPU_THREAD_COUNT = str(min(8, os.cpu_count() or 4))
os.environ["XDG_CACHE_HOME"] = _CACHE_DIR
os.environ["HF_HOME"] = _HF_CACHE_DIR
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["OMP_NUM_THREADS"] = _DEFAULT_CPU_THREAD_COUNT
os.environ["VECLIB_MAXIMUM_THREADS"] = os.environ["OMP_NUM_THREADS"]
os.environ["NUMEXPR_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]
os.environ["MPLCONFIGDIR"] = _MPLCONFIGDIR
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from transformers import (
        AutoFeatureExtractor,
        Wav2Vec2ForSequenceClassification,
        get_linear_schedule_with_warmup,
    )
except ModuleNotFoundError as exc:
    if exc.name in {"transformers", "huggingface_hub", "tokenizers", "safetensors"}:
        raise RuntimeError(
            "Missing wav2vec2 fine-tuning dependencies in fix-paper1. "
            "Update the environment before running this script. "
            "Recommended command: conda env update -n fix-paper1 -f environment.yml"
        ) from exc
    raise

TQDM_MININTERVAL = float(os.environ.get("TQDM_MININTERVAL", 2.0))
TQDM_DISABLE = os.environ.get("DISABLE_TQDM", "0") == "1"
TQDM_VERBOSE = os.environ.get("TQDM_VERBOSE", "0") == "1"


class SilentProgress:
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, *args, **kwargs):
        pass


def progress(iterable, desc=None, total=None, unit="it", leave=True):
    if not leave and not TQDM_VERBOSE:
        return SilentProgress(iterable)
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        leave=leave,
        dynamic_ncols=True,
        mininterval=TQDM_MININTERVAL,
        disable=TQDM_DISABLE,
    )


def parse_int_list_env(env_name, default_values):
    raw_value = os.environ.get(env_name)
    if not raw_value:
        return list(default_values)
    values = []
    for item in raw_value.replace(";", ",").split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError(f"{env_name} does not contain any valid seed.")
    return values


def parse_bool_env(env_name, default=False):
    raw_value = os.environ.get(env_name)
    if raw_value is None:
        return bool(default)
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def slugify(value):
    replacements = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "à": "a", "è": "e", "ì": "i", "ò": "o", "ù": "u",
        "ä": "a", "ë": "e", "ï": "i", "ö": "o", "ü": "u",
        "ñ": "n", "ç": "c",
    }
    value = str(value).strip().lower()
    for old, new in replacements.items():
        value = value.replace(old, new)
    chars = []
    previous_sep = False
    for char in value:
        if char.isalnum():
            chars.append(char)
            previous_sep = False
        elif not previous_sep:
            chars.append("-")
            previous_sep = True
    return "".join(chars).strip("-") or "sin-etiqueta"


def infer_recording_id(segment_file):
    return re.sub(r"_ev\d+$", "", os.path.splitext(segment_file)[0])


def resolve_device_name():
    requested = os.environ.get("WAV2VEC2_DEVICE", "auto").strip().lower()
    mps_available = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()
    if requested == "auto":
        if mps_available:
            return "mps"
        if cuda_available:
            return "cuda"
        return "cpu"
    if requested == "mps":
        if not mps_available:
            raise RuntimeError("WAV2VEC2_DEVICE=mps but MPS is not available in this environment.")
        return "mps"
    if requested == "cuda":
        if not cuda_available:
            raise RuntimeError("WAV2VEC2_DEVICE=cuda but CUDA is not available in this environment.")
        return "cuda"
    if requested == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported WAV2VEC2_DEVICE: {requested}")


DATASET_NAME = os.environ.get("DATASET_NAME", "0 SPRSound")
DATASET_LABEL = os.environ.get("DATASET_LABEL", "sprsound")
TRAIN_DATASET_LABEL = os.environ.get("TRAIN_DATASET_LABEL", DATASET_LABEL)
VALIDATION_DATASET_LABEL = os.environ.get("VALIDATION_DATASET_LABEL", TRAIN_DATASET_LABEL)
TEST_DATASET_LABEL = os.environ.get("TEST_DATASET_LABEL", DATASET_LABEL)
DATASET_PROTOCOL = os.environ.get(
    "DATASET_PROTOCOL",
    f"{slugify(TRAIN_DATASET_LABEL)}_to_{slugify(TEST_DATASET_LABEL)}",
)
EXPERIMENT_BLOCK = os.environ.get("EXPERIMENT_BLOCK", "03_wav2vec2_finetune")
MODEL_FAMILY = os.environ.get("MODEL_FAMILY", "wav2vec2")
MODEL_DISPLAY_NAME = os.environ.get("MODEL_DISPLAY_NAME", "Wav2Vec2")
BACKBONE_NAME = os.environ.get("BACKBONE_NAME", "facebook_wav2vec2_base")
TRAINING_MODE = os.environ.get("TRAINING_MODE", "finetuned")
CLASSIFIER_TYPE = os.environ.get("CLASSIFIER_TYPE", "sequence_classification_head")
EMBEDDING_MODE = os.environ.get("EMBEDDING_MODE", "not_applicable_end_to_end")
FINETUNING_SCOPE = os.environ.get("FINETUNING_SCOPE", "full")
EXPERIMENT_NOTES = os.environ.get("EXPERIMENT_NOTES", "")
CHECKPOINT_SELECTION_METRIC = os.environ.get("CHECKPOINT_SELECTION_METRIC", "validation_auc_roc_then_auc_pr")
SPLIT_SEED = int(os.environ.get("SPLIT_SEED", 42))
SEEDS = parse_int_list_env("EXPERIMENT_SEEDS", [42])
SEED_LIST_LABEL = "-".join(str(seed) for seed in SEEDS)
TRAINING_SEEDS_TAG = f"seeds_{SEED_LIST_LABEL}"
REPORT_THRESHOLD_STRATEGY = os.environ.get("REPORT_THRESHOLD_STRATEGY", "val_best_f1").strip()

MODEL_ID = os.environ.get("WAV2VEC2_MODEL_ID", "facebook/wav2vec2-base")
SAMPLE_RATE = int(os.environ.get("WAV2VEC2_SAMPLE_RATE", 16000))
DEVICE_NAME = resolve_device_name()
TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", 4))
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", 8))
GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", 1))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", 8))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 1e-5))
# R1: lr propio para la cabeza (projector + classifier, inicializados al azar); por defecto igual al global.
HEAD_LEARNING_RATE = float(os.environ.get("HEAD_LEARNING_RATE", LEARNING_RATE))
# R1: subconjuntos estratificados SOLO para pruebas cortas (0 = sin limite).
MAX_TRAIN_SAMPLES = int(os.environ.get("MAX_TRAIN_SAMPLES", 0))
MAX_TEST_SAMPLES = int(os.environ.get("MAX_TEST_SAMPLES", 0))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 1e-2))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", 0.1))
EARLY_STOPPING_PATIENCE = int(os.environ.get("EARLY_STOPPING_PATIENCE", 3))
MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", 1.0))
DATALOADER_NUM_WORKERS = int(os.environ.get("DATALOADER_NUM_WORKERS", 0))
LOCAL_FILES_ONLY = parse_bool_env("HF_LOCAL_FILES_ONLY", False)
USE_SAFETENSORS = parse_bool_env("WAV2VEC2_USE_SAFETENSORS", True)
FREEZE_FEATURE_ENCODER = parse_bool_env("FREEZE_FEATURE_ENCODER", False)
ENABLE_GRADIENT_CHECKPOINTING = parse_bool_env("ENABLE_GRADIENT_CHECKPOINTING", False)
WAVEFORM_CACHE_SIZE = max(0, int(os.environ.get("WAVEFORM_CACHE_SIZE", 4096)))

CLASS_WEIGHT_MODE = os.environ.get("CLASS_WEIGHT_MODE", "none").strip().lower()
WEIGHT1 = float(os.environ.get("CLASS_WEIGHT_NEGATIVE", 1.0))
WEIGHT2 = float(os.environ.get("CLASS_WEIGHT_POSITIVE", 1.0))

DEFAULT_SEGMENTED_BASE_DIR = os.path.join(
    _PROJECT_DIR,
    "ficheros",
    "1 Segmentados_WAV",
    DATASET_NAME,
    f"seed_{SPLIT_SEED}",
)
TRAIN_SEGMENTED_WAV_DIR = os.environ.get(
    "TRAIN_SEGMENTED_WAV_DIR",
    os.path.join(DEFAULT_SEGMENTED_BASE_DIR, "segmentado_train"),
)
VALIDATION_SEGMENTED_WAV_DIR = os.environ.get(
    "VALIDATION_SEGMENTED_WAV_DIR",
    os.path.join(DEFAULT_SEGMENTED_BASE_DIR, "segmentado_validation"),
)
TEST_SEGMENTED_WAV_DIR = os.environ.get(
    "TEST_SEGMENTED_WAV_DIR",
    os.path.join(DEFAULT_SEGMENTED_BASE_DIR, "segmentado_test"),
)

INPUT_BASE_DIR = os.path.join(_PROJECT_DIR, "ficheros", DATASET_NAME)
SPLIT_MANIFEST_DIR = os.path.join(_PROJECT_DIR, "ficheros", "splits", DATASET_NAME)
DEFAULT_SPLIT_MANIFEST_JSON = os.path.join(SPLIT_MANIFEST_DIR, f"splits_patient_independent_seed{SPLIT_SEED}.json")
DEFAULT_SPLIT_MANIFEST_CSV = os.path.join(SPLIT_MANIFEST_DIR, f"splits_patient_independent_seed{SPLIT_SEED}.csv")
SPLIT_MANIFEST_JSON = os.environ.get("SPLIT_MANIFEST_JSON", DEFAULT_SPLIT_MANIFEST_JSON)
SPLIT_MANIFEST_CSV = os.environ.get("SPLIT_MANIFEST_CSV", DEFAULT_SPLIT_MANIFEST_CSV)
TRAIN_SPLIT_MANIFEST_JSON = os.environ.get("TRAIN_SPLIT_MANIFEST_JSON", SPLIT_MANIFEST_JSON)
TRAIN_SPLIT_MANIFEST_CSV = os.environ.get("TRAIN_SPLIT_MANIFEST_CSV", SPLIT_MANIFEST_CSV)
VALIDATION_SPLIT_MANIFEST_JSON = os.environ.get("VALIDATION_SPLIT_MANIFEST_JSON", TRAIN_SPLIT_MANIFEST_JSON)
VALIDATION_SPLIT_MANIFEST_CSV = os.environ.get("VALIDATION_SPLIT_MANIFEST_CSV", TRAIN_SPLIT_MANIFEST_CSV)
TEST_SPLIT_MANIFEST_JSON = os.environ.get("TEST_SPLIT_MANIFEST_JSON", SPLIT_MANIFEST_JSON)
TEST_SPLIT_MANIFEST_CSV = os.environ.get("TEST_SPLIT_MANIFEST_CSV", SPLIT_MANIFEST_CSV)

FECHA_EJECUCION = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ID = os.environ.get("RUN_ID", f"run_{FECHA_EJECUCION}")
MODEL_SETUP_TAG = "_".join(
    slugify(part)
    for part in (MODEL_FAMILY, BACKBONE_NAME, TRAINING_MODE, CLASSIFIER_TYPE, FINETUNING_SCOPE)
    if str(part).strip() and str(part).strip().lower() != "none"
)
MODEL_SETUP_TAG = MODEL_SETUP_TAG or slugify(MODEL_FAMILY)
RUN_BASE_NAME = (
    f"{slugify(MODEL_FAMILY)}_{slugify(TRAINING_MODE)}_{slugify(CLASSIFIER_TYPE)}_"
    f"splitSeed{SPLIT_SEED}_{TRAINING_SEEDS_TAG}_{RUN_ID}"
)
MODEL_OUTPUT_DIR = os.path.join(
    _PROJECT_DIR,
    "modelos",
    slugify(EXPERIMENT_BLOCK),
    slugify(DATASET_PROTOCOL),
    slugify(MODEL_FAMILY),
    MODEL_SETUP_TAG,
    f"split_seed_{SPLIT_SEED}",
    RUN_ID,
)
SUMMARY_OUTPUT_DIR = os.path.join(MODEL_OUTPUT_DIR, "summaries")

EXPERIMENT_METADATA = {
    "Experiment_Block": EXPERIMENT_BLOCK,
    "Dataset_Protocol": DATASET_PROTOCOL,
    "Train_Dataset": TRAIN_DATASET_LABEL,
    "Validation_Dataset": VALIDATION_DATASET_LABEL,
    "Test_Dataset": TEST_DATASET_LABEL,
    "Model_Family": MODEL_FAMILY,
    "Model_Display_Name": MODEL_DISPLAY_NAME,
    "Backbone": BACKBONE_NAME,
    "Training_Mode": TRAINING_MODE,
    "Embedding_Mode": EMBEDDING_MODE,
    "Classifier_Type": CLASSIFIER_TYPE,
    "Finetuning_Scope": FINETUNING_SCOPE,
    "Experiment_Notes": EXPERIMENT_NOTES,
    "Run_ID": RUN_ID,
    "Run_Base_Name": RUN_BASE_NAME,
}

EVENT_TO_BINARY = {
    "Normal": 0,
    "Crackle": 1,
    "Rhonchi": 1,
    "Wheeze": 1,
    "Stridor": 1,
    "Coarse_Crackle": 1,
    "Fine_Crackle": 1,
    "Wheeze_Crackle": 1,
}


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_seed_output_dirs(seed):
    seed_root = os.path.join(MODEL_OUTPUT_DIR, f"seed_{seed}")
    dirs = {
        "root": seed_root,
        "checkpoints": os.path.join(seed_root, "checkpoints"),
        "reports": os.path.join(seed_root, "reports"),
        "predictions": os.path.join(seed_root, "predictions"),
        "figures": os.path.join(seed_root, "figures"),
        "calibration": os.path.join(seed_root, "calibration"),
        "history": os.path.join(seed_root, "history"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def build_run_manifest():
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_dir": _PROJECT_DIR,
        "current_experiment": EXPERIMENT_METADATA,
        "notes": EXPERIMENT_NOTES,
        "paths": {
            "model_output_dir": MODEL_OUTPUT_DIR,
            "summary_output_dir": SUMMARY_OUTPUT_DIR,
            "split_manifest_json": SPLIT_MANIFEST_JSON,
            "split_manifest_csv": SPLIT_MANIFEST_CSV,
            "split_manifests_by_split": {
                "train": {"json": TRAIN_SPLIT_MANIFEST_JSON, "csv": TRAIN_SPLIT_MANIFEST_CSV},
                "validation": {"json": VALIDATION_SPLIT_MANIFEST_JSON, "csv": VALIDATION_SPLIT_MANIFEST_CSV},
                "test": {"json": TEST_SPLIT_MANIFEST_JSON, "csv": TEST_SPLIT_MANIFEST_CSV},
            },
            "segmented_wavs_by_split": {
                "train": TRAIN_SEGMENTED_WAV_DIR,
                "validation": VALIDATION_SEGMENTED_WAV_DIR,
                "test": TEST_SEGMENTED_WAV_DIR,
            },
        },
        "training": {
            "model_id": MODEL_ID,
            "sample_rate": SAMPLE_RATE,
            "device": DEVICE_NAME,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "head_learning_rate": HEAD_LEARNING_RATE,
            "max_train_samples": MAX_TRAIN_SAMPLES,
            "max_test_samples": MAX_TEST_SAMPLES,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "max_grad_norm": MAX_GRAD_NORM,
            "class_weight_mode": CLASS_WEIGHT_MODE,
            "weight1": WEIGHT1,
            "weight2": WEIGHT2,
            "freeze_feature_encoder": FREEZE_FEATURE_ENCODER,
            "enable_gradient_checkpointing": ENABLE_GRADIENT_CHECKPOINTING,
            "checkpoint_selection_metric": CHECKPOINT_SELECTION_METRIC,
            "report_threshold_strategy": REPORT_THRESHOLD_STRATEGY,
            "training_seeds": SEEDS,
            "dataloader_num_workers": DATALOADER_NUM_WORKERS,
            "waveform_cache_size": WAVEFORM_CACHE_SIZE,
            "local_files_only": LOCAL_FILES_ONLY,
            "use_safetensors": USE_SAFETENSORS,
        },
    }


def write_run_manifest():
    manifest_path = os.path.join(MODEL_OUTPUT_DIR, "run_manifest.json")
    write_json(manifest_path, build_run_manifest())
    print(f">>> wav2vec2 fine-tuning run manifest saved to: {manifest_path}")


def write_seed_manifest(seed, seed_dirs, dataset_sizes, training_summary=None):
    manifest = build_run_manifest()
    manifest.update(
        {
            "seed": seed,
            "seed_output_dirs": seed_dirs,
            "dataset_sizes": dataset_sizes,
            "training_summary": training_summary or {},
        }
    )
    manifest_path = os.path.join(seed_dirs["root"], "seed_manifest.json")
    write_json(manifest_path, manifest)
    print(f">>> Seed manifest saved to: {manifest_path}")


def configure_torch_runtime():
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assert_segmented_wavs_ready():
    required = {
        "train": TRAIN_SEGMENTED_WAV_DIR,
        "validation": VALIDATION_SEGMENTED_WAV_DIR,
        "test": TEST_SEGMENTED_WAV_DIR,
    }
    missing = []
    for split_name, split_dir in required.items():
        if not os.path.isdir(split_dir):
            missing.append(f"{split_name}:{split_dir}")
            continue
        if not glob.glob(os.path.join(split_dir, "*", "*.wav")):
            missing.append(f"{split_name}:{split_dir}")
    if missing:
        raise FileNotFoundError(
            "Missing segmented WAVs for wav2vec2 fine-tuning: "
            + ", ".join(missing)
            + ". Run the corresponding CNN preprocess first."
        )


def load_segment_paths(split_dir):
    return sorted(glob.glob(os.path.join(split_dir, "*", "*.wav")))


def infer_binary_label(segment_path):
    label_name = os.path.basename(os.path.dirname(segment_path))
    if label_name not in EVENT_TO_BINARY:
        raise ValueError(f"Unsupported segment label for wav2vec2 fine-tuning: {label_name}")
    return label_name, EVENT_TO_BINARY[label_name]


def infer_source_dataset(segment_path):
    basename = os.path.basename(segment_path)
    if basename.startswith("sprsound__"):
        return "sprsound"
    if basename.startswith("icbhi__"):
        return "icbhi"
    return DATASET_LABEL


def _load_waveform_uncached(segment_path):
    waveform, _ = librosa.load(segment_path, sr=SAMPLE_RATE, mono=True)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0:
        raise ValueError("Empty waveform.")
    return waveform


if WAVEFORM_CACHE_SIZE > 0:
    _load_waveform_cached = lru_cache(maxsize=WAVEFORM_CACHE_SIZE)(_load_waveform_uncached)
else:
    _load_waveform_cached = _load_waveform_uncached


def load_waveform(segment_path):
    return _load_waveform_cached(segment_path)


def compute_min_input_samples(model):
    length_fn = getattr(model, "_get_feat_extract_output_lengths", None)
    if length_fn is None and hasattr(model, "wav2vec2"):
        length_fn = getattr(model.wav2vec2, "_get_feat_extract_output_lengths", None)
    if length_fn is None:
        raise RuntimeError("Could not resolve wav2vec2 feature-length helper.")
    candidate_lengths = torch.arange(1, SAMPLE_RATE + 1, dtype=torch.long)
    output_lengths = length_fn(candidate_lengths)
    valid_positions = (output_lengths >= 1).nonzero(as_tuple=False)
    if valid_positions.numel() == 0:
        raise RuntimeError("Could not compute the minimum input length for wav2vec2.")
    return int(valid_positions[0].item() + 1)


def pad_waveform(waveform, min_input_samples):
    if waveform.shape[0] >= min_input_samples:
        return waveform, int(waveform.shape[0]), False
    padded = np.pad(waveform, (0, min_input_samples - waveform.shape[0]), mode="constant")
    return padded.astype(np.float32, copy=False), int(min_input_samples), True


def use_attention_mask_for_model(model):
    feat_extract_norm = str(getattr(model.config, "feat_extract_norm", "")).strip().lower()
    return feat_extract_norm == "layer"


class AudioSegmentDataset(Dataset):
    def __init__(self, split_name, split_dir):
        self.split_name = split_name
        self.split_dir = split_dir
        self.records = []
        for segment_path in load_segment_paths(split_dir):
            label_name, label_value = infer_binary_label(segment_path)
            segment_file = os.path.basename(segment_path)
            self.records.append(
                {
                    "split": split_name,
                    "sample_path": segment_path,
                    "segment_file": segment_file,
                    "recording_id": infer_recording_id(segment_file),
                    "label_name": label_name,
                    "label_binary": label_value,
                    "source_dataset": infer_source_dataset(segment_path),
                }
            )
        limit = {"train": MAX_TRAIN_SAMPLES, "test": MAX_TEST_SAMPLES}.get(split_name, 0)
        if limit and limit < len(self.records):
            # Submuestreo estratificado por clase con semilla fija (solo para pruebas cortas).
            rng = np.random.default_rng(20261029)
            labels = np.asarray([r["label_binary"] for r in self.records])
            keep = []
            for cls in (0, 1):
                idx = np.flatnonzero(labels == cls)
                n_cls = int(round(limit * len(idx) / len(labels)))
                keep.extend(rng.choice(idx, size=min(n_cls, len(idx)), replace=False).tolist())
            self.records = [self.records[i] for i in sorted(keep)]
        self.labels = np.asarray([record["label_binary"] for record in self.records], dtype=np.int64)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx].copy()
        waveform = load_waveform(record["sample_path"])
        record["waveform"] = waveform
        record["input_samples_original"] = int(waveform.shape[0])
        record["duration_seconds_resampled"] = float(waveform.shape[0] / SAMPLE_RATE)
        return record


class AudioBatchCollator:
    def __init__(self, feature_extractor, min_input_samples):
        self.feature_extractor = feature_extractor
        self.min_input_samples = min_input_samples

    def __call__(self, batch):
        waveforms = []
        labels = []
        metadata_rows = []
        for record in batch:
            waveform, effective_samples, was_padded = pad_waveform(record["waveform"], self.min_input_samples)
            waveforms.append(waveform)
            labels.append(record["label_binary"])
            metadata_rows.append(
                {
                    "split": record["split"],
                    "sample_path": record["sample_path"],
                    "segment_file": record["segment_file"],
                    "recording_id": record["recording_id"],
                    "label_name": record["label_name"],
                    "label_binary": record["label_binary"],
                    "source_dataset": record["source_dataset"],
                    "duration_seconds_resampled": record["duration_seconds_resampled"],
                    "input_samples_original": record["input_samples_original"],
                    "input_samples_effective": effective_samples,
                    "input_was_padded": was_padded,
                }
            )
        encoded = self.feature_extractor(
            waveforms,
            sampling_rate=SAMPLE_RATE,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_values": encoded["input_values"],
            "attention_mask": encoded.get("attention_mask"),
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "metadata_rows": metadata_rows,
        }


def build_dataloader(dataset, collator, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=DATALOADER_NUM_WORKERS,
        collate_fn=collator,
        drop_last=False,
        pin_memory=DEVICE_NAME == "cuda",
        persistent_workers=DATALOADER_NUM_WORKERS > 0,
    )


def load_model_and_extractor():
    configure_torch_runtime()
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, local_files_only=LOCAL_FILES_ONLY)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=2,
        problem_type="single_label_classification",
        label2id={"healthy": 0, "adventitious": 1},
        id2label={0: "Healthy", 1: "Adventitious"},
        local_files_only=LOCAL_FILES_ONLY,
        use_safetensors=USE_SAFETENSORS,
        ignore_mismatched_sizes=True,
    )
    if ENABLE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
    if FREEZE_FEATURE_ENCODER:
        if hasattr(model, "freeze_feature_encoder"):
            model.freeze_feature_encoder()
        elif hasattr(model, "freeze_feature_extractor"):
            model.freeze_feature_extractor()
    model.to(torch.device(DEVICE_NAME))
    return feature_extractor, model


def move_batch_to_device(batch, pass_attention_mask_to_model):
    model_inputs = {
        "input_values": batch["input_values"].to(torch.device(DEVICE_NAME)),
    }
    if pass_attention_mask_to_model and batch.get("attention_mask") is not None:
        model_inputs["attention_mask"] = batch["attention_mask"].to(torch.device(DEVICE_NAME))
    labels = batch["labels"].to(torch.device(DEVICE_NAME))
    return model_inputs, labels


def resolve_class_weights(train_labels):
    if CLASS_WEIGHT_MODE == "none":
        return None, {}
    if CLASS_WEIGHT_MODE == "manual":
        weights = torch.tensor([WEIGHT1, WEIGHT2], dtype=torch.float32, device=torch.device(DEVICE_NAME))
        return weights, {"negative": WEIGHT1, "positive": WEIGHT2}
    if CLASS_WEIGHT_MODE == "balanced":
        train_labels = np.asarray(train_labels, dtype=np.int64)
        total = float(len(train_labels))
        negatives = max(1.0, float(np.sum(train_labels == 0)))
        positives = max(1.0, float(np.sum(train_labels == 1)))
        negative_weight = total / (2.0 * negatives)
        positive_weight = total / (2.0 * positives)
        weights = torch.tensor([negative_weight, positive_weight], dtype=torch.float32, device=torch.device(DEVICE_NAME))
        return weights, {"negative": negative_weight, "positive": positive_weight}
    raise ValueError(f"Unsupported CLASS_WEIGHT_MODE: {CLASS_WEIGHT_MODE}")


def compute_loss(logits, labels, class_weights):
    return F.cross_entropy(logits, labels, weight=class_weights)


def clip_probabilities(probabilities, eps=1e-6):
    return np.clip(np.asarray(probabilities, dtype=np.float64), eps, 1.0 - eps)


def probability_to_logit(probabilities):
    probabilities = clip_probabilities(probabilities)
    return np.log(probabilities / (1.0 - probabilities))


def sigmoid_np(values):
    values = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-values))


def compute_split_metrics(y_true, probabilities, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = clip_probabilities(probabilities)
    threshold = float(threshold)
    predictions = (probabilities >= threshold).astype(int)
    report = classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=["Healthy (Normal)", "Adventitious"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, probabilities)
        roc_auc = auc(fpr, tpr)
        pr_auc = average_precision_score(y_true, probabilities)
        logloss = log_loss(y_true, probabilities, labels=[0, 1])
    else:
        roc_auc, pr_auc, logloss = np.nan, np.nan, np.nan
    return {
        "y_true": y_true,
        "probabilities": probabilities,
        "predictions": predictions,
        "classification_report": report,
        "confusion_matrix": cm,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "auc_roc": roc_auc,
        "auc_pr": pr_auc,
        "accuracy": accuracy_score(y_true, predictions),
        "logloss": logloss,
        "f1_adv": f1_score(y_true, predictions, zero_division=0),
        "recall_adv": recall_score(y_true, predictions, zero_division=0),
        "precision_adv": precision_score(y_true, predictions, zero_division=0),
        "threshold": threshold,
    }


def apply_decision_threshold(metrics, threshold, threshold_strategy):
    updated = compute_split_metrics(metrics["y_true"], metrics["probabilities"], threshold=threshold)
    updated["loss"] = metrics.get("loss", np.nan)
    updated["metadata"] = metrics.get("metadata", pd.DataFrame()).copy()
    updated["threshold_strategy"] = threshold_strategy
    return updated


def resolve_report_threshold(y_true, probabilities):
    thresholds = select_thresholds_from_validation(y_true, probabilities)
    if REPORT_THRESHOLD_STRATEGY in thresholds:
        strategy = REPORT_THRESHOLD_STRATEGY
    else:
        fallback_order = ["val_best_f1", "val_best_balanced_accuracy", "fixed_0_5"]
        strategy = next((name for name in fallback_order if name in thresholds), "fixed_0_5")
        print(
            f">>> Requested REPORT_THRESHOLD_STRATEGY={REPORT_THRESHOLD_STRATEGY!r} is not available. "
            f"Using {strategy!r} instead."
        )
    return strategy, float(thresholds[strategy]), thresholds


def evaluation_key(metrics):
    roc_auc = metrics["auc_roc"] if np.isfinite(metrics["auc_roc"]) else -np.inf
    pr_auc = metrics["auc_pr"] if np.isfinite(metrics["auc_pr"]) else -np.inf
    loss_value = metrics["loss"] if np.isfinite(metrics["loss"]) else np.inf
    return (roc_auc, pr_auc, -loss_value)


def evaluate_model(model, dataloader, split_name, class_weights, pass_attention_mask_to_model, seed):
    model.eval()
    losses = []
    probabilities = []
    labels = []
    metadata_rows = []
    for batch in progress(dataloader, desc=f"{split_name.title()} seed {seed}", unit="batch", leave=False):
        model_inputs, batch_labels = move_batch_to_device(batch, pass_attention_mask_to_model)
        with torch.inference_mode():
            outputs = model(**model_inputs)
            loss = compute_loss(outputs.logits, batch_labels, class_weights)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
        losses.append(float(loss.detach().cpu().item()))
        probabilities.append(probs.detach().cpu().numpy())
        labels.append(batch_labels.detach().cpu().numpy())
        metadata_rows.extend(batch["metadata_rows"])
    y_true = np.concatenate(labels, axis=0)
    probs = np.concatenate(probabilities, axis=0)
    metrics = compute_split_metrics(y_true, probs)
    metrics["loss"] = float(np.mean(losses)) if losses else np.nan
    metrics["metadata"] = pd.DataFrame(metadata_rows)
    return metrics


def train_one_epoch(model, dataloader, optimizer, scheduler, class_weights, pass_attention_mask_to_model, seed, epoch_idx):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_examples = 0
    running_loss = 0.0
    running_correct = 0
    train_iterator = progress(
        dataloader,
        desc=f"Train seed {seed} | epoch {epoch_idx}/{NUM_EPOCHS}",
        unit="batch",
        leave=True,
    )
    for step_idx, batch in enumerate(train_iterator, start=1):
        model_inputs, labels = move_batch_to_device(batch, pass_attention_mask_to_model)
        outputs = model(**model_inputs)
        loss = compute_loss(outputs.logits, labels, class_weights)
        batch_size = int(labels.shape[0])
        running_loss += float(loss.detach().cpu().item()) * batch_size
        predictions = outputs.logits.argmax(dim=-1)
        running_correct += int((predictions == labels).sum().detach().cpu().item())
        total_examples += batch_size
        (loss / max(1, GRAD_ACCUM_STEPS)).backward()

        should_step = (step_idx % max(1, GRAD_ACCUM_STEPS) == 0) or (step_idx == len(dataloader))
        if should_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        if step_idx == 1 or step_idx % 25 == 0 or step_idx == len(dataloader):
            train_iterator.set_postfix(
                loss=f"{(running_loss / max(1, total_examples)):.4f}",
                acc=f"{(running_correct / max(1, total_examples)):.4f}",
            )
    return {
        "loss": running_loss / max(1, total_examples),
        "accuracy": running_correct / max(1, total_examples),
    }


def save_analysis_figure(y_true, y_pred, probabilities, figure_file, threshold=0.5, threshold_strategy="fixed_0_5"):
    target_names = ["Healthy (Normal)", "Adventitious"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, probabilities)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, probabilities)
        roc_auc = auc(fpr, tpr)
        pr_auc = average_precision_score(y_true, probabilities)
    else:
        fpr, tpr = np.array([]), np.array([])
        precision_curve, recall_curve = np.array([]), np.array([])
        roc_auc, pr_auc = np.nan, np.nan

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Confusion Matrix ({threshold_strategy}, thr={threshold:.4f})")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    plt.subplot(1, 3, 2)
    if len(fpr):
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.legend(loc="lower right")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.subplot(1, 3, 3)
    if len(precision_curve):
        plt.plot(recall_curve, precision_curve, color="darkgreen", lw=2, label=f"PR AUC = {pr_auc:.2f}")
        plt.legend(loc="lower left")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    os.makedirs(os.path.dirname(figure_file), exist_ok=True)
    plt.savefig(figure_file, dpi=300)
    plt.close()


def export_split_report(split_name, split_metrics, metadata, excel_file, predictions_file, figure_file, seed=None):
    print("\n" + "=" * 30)
    print(f"GENERATING FINAL REPORT: {split_name.upper()}")
    print("=" * 30)
    print(
        classification_report(
            split_metrics["y_true"],
            split_metrics["predictions"],
            labels=[0, 1],
            target_names=["Healthy (Normal)", "Adventitious"],
            zero_division=0,
        )
    )

    report_dict = split_metrics["classification_report"]
    cm = split_metrics["confusion_matrix"]
    df_report = pd.DataFrame(report_dict).transpose()
    df_cm = pd.DataFrame(
        cm,
        index=["True_Healthy", "True_Adventitious"],
        columns=["Pred_Healthy", "Pred_Adventitious"],
    )
    df_predictions = metadata.copy()
    df_predictions["run_id"] = RUN_ID
    df_predictions["experiment_block"] = EXPERIMENT_BLOCK
    df_predictions["dataset_protocol"] = DATASET_PROTOCOL
    df_predictions["model_family"] = MODEL_FAMILY
    df_predictions["training_mode"] = TRAINING_MODE
    df_predictions["classifier_type"] = CLASSIFIER_TYPE
    df_predictions["seed"] = seed
    df_predictions["split"] = split_name
    df_predictions["y_true"] = split_metrics["y_true"]
    df_predictions["prob_adventitious"] = split_metrics["probabilities"]
    df_predictions["y_pred"] = split_metrics["predictions"]
    df_predictions["threshold"] = split_metrics.get("threshold", 0.5)
    df_predictions["threshold_strategy"] = split_metrics.get("threshold_strategy", "fixed_0_5")
    df_predictions["y_pred_threshold_0_5"] = (split_metrics["probabilities"] >= 0.5).astype(int)
    os.makedirs(os.path.dirname(predictions_file), exist_ok=True)
    df_predictions.to_csv(predictions_file, index=False)
    print(f">>> Predictions and probabilities saved to: {predictions_file}")

    result_row = {
        "Fecha": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        **EXPERIMENT_METADATA,
        "Seed": seed,
        "Split_Seed": SPLIT_SEED,
        "Training_Seeds": SEED_LIST_LABEL,
        "Split": split_name,
        "Modelo": MODEL_DISPLAY_NAME,
        "Checkpoint_Selection": CHECKPOINT_SELECTION_METRIC,
        "Threshold_Strategy": split_metrics.get("threshold_strategy", "fixed_0_5"),
        "Threshold": round(float(split_metrics.get("threshold", 0.5)), 6),
        "Class_Weight_Mode": CLASS_WEIGHT_MODE,
        "Weight1": WEIGHT1,
        "Weight2": WEIGHT2,
        "AUC_ROC": round(split_metrics["auc_roc"], 4) if not np.isnan(split_metrics["auc_roc"]) else np.nan,
        "AUC_PR": round(split_metrics["auc_pr"], 4) if not np.isnan(split_metrics["auc_pr"]) else np.nan,
        "F1-Score_Adv": round(report_dict["Adventitious"]["f1-score"], 4),
        "Recall_Adv": round(report_dict["Adventitious"]["recall"], 4),
        "Precision_Adv": round(report_dict["Adventitious"]["precision"], 4),
        "TN": int(split_metrics["tn"]),
        "FP": int(split_metrics["fp"]),
        "FN": int(split_metrics["fn"]),
        "TP": int(split_metrics["tp"]),
    }
    print(result_row)

    df_result_row = pd.DataFrame([result_row])
    sheet_name = split_name[:31]
    gap = 3
    os.makedirs(os.path.dirname(excel_file), exist_ok=True)
    if not os.path.exists(excel_file):
        writer_mode = {"mode": "w"}
    else:
        writer_mode = {"mode": "a", "if_sheet_exists": "replace"}
    with pd.ExcelWriter(excel_file, engine="openpyxl", **writer_mode) as writer:
        df_report.to_excel(writer, sheet_name=sheet_name)
        start_row_cm = len(df_report) + gap
        df_cm.to_excel(writer, sheet_name=sheet_name, startrow=start_row_cm)
        start_row_summary = start_row_cm + len(df_cm) + gap
        df_result_row.to_excel(writer, sheet_name=sheet_name, startrow=start_row_summary)
    print(f">>> Excel report saved to: {excel_file}")

    save_analysis_figure(
        split_metrics["y_true"],
        split_metrics["predictions"],
        split_metrics["probabilities"],
        figure_file,
        threshold=split_metrics.get("threshold", 0.5),
        threshold_strategy=split_metrics.get("threshold_strategy", "fixed_0_5"),
    )
    print(f">>> Analysis figure saved to: {figure_file}")
    return result_row


def select_thresholds_from_validation(y_true, scores):
    y_true = np.asarray(y_true, dtype=int)
    scores = clip_probabilities(scores)
    thresholds = {"fixed_0_5": 0.5}
    candidate_thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], scores)))
    best_f1, best_f1_threshold = -1.0, 0.5
    best_balanced, best_balanced_threshold = -1.0, 0.5
    recall_target_threshold = None
    recall_target_specificity = -1.0
    for threshold in candidate_thresholds:
        y_pred = (scores >= threshold).astype(int)
        current_f1 = f1_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn + 1e-9)
        specificity = tn / (tn + fp + 1e-9)
        balanced = 0.5 * (sensitivity + specificity)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_f1_threshold = float(threshold)
        if balanced > best_balanced:
            best_balanced = balanced
            best_balanced_threshold = float(threshold)
        if sensitivity >= 0.80 and specificity > recall_target_specificity:
            recall_target_threshold = float(threshold)
            recall_target_specificity = specificity
    thresholds["val_best_f1"] = best_f1_threshold
    thresholds["val_best_balanced_accuracy"] = best_balanced_threshold
    if recall_target_threshold is not None:
        thresholds["val_recall_at_least_0_80"] = recall_target_threshold
    if len(np.unique(y_true)) == 2:
        fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
        finite = np.isfinite(roc_thresholds)
        if finite.any():
            idx = np.argmax((tpr - fpr)[finite])
            thresholds["val_youden"] = float(roc_thresholds[finite][idx])
    return thresholds


def binary_metrics_for_threshold(y_true, scores, threshold, split_name, calibration_name, threshold_name, seed=None, extra=None):
    y_true = np.asarray(y_true, dtype=int)
    scores = clip_probabilities(scores)
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        pr_auc = average_precision_score(y_true, scores)
        logloss = log_loss(y_true, scores, labels=[0, 1])
    else:
        roc_auc, pr_auc, logloss = np.nan, np.nan, np.nan

    row = {
        "Fecha": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        **EXPERIMENT_METADATA,
        "Seed": seed,
        "Split": split_name,
        "Calibration": calibration_name,
        "Threshold_Strategy": threshold_name,
        "Threshold": float(threshold),
        "N": int(len(y_true)),
        "Positives": int(np.sum(y_true == 1)),
        "Negatives": int(np.sum(y_true == 0)),
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC_ROC": roc_auc,
        "AUC_PR": pr_auc,
        "Brier": brier_score_loss(y_true, scores),
        "LogLoss": logloss,
        "F1_Adv": f1_score(y_true, y_pred, zero_division=0),
        "Recall_Adv": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp + 1e-9),
        "Precision_Adv": precision_score(y_true, y_pred, zero_division=0),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }
    if extra:
        row.update(extra)
    return row


def fit_temperature_from_validation(y_true, probabilities):
    y_true = np.asarray(y_true, dtype=int)
    logits = probability_to_logit(probabilities)
    candidate_temperatures = np.concatenate([np.linspace(0.20, 2.00, 91), np.linspace(2.05, 8.00, 120)])
    losses = []
    for temperature in candidate_temperatures:
        calibrated = sigmoid_np(logits / temperature)
        losses.append(log_loss(y_true, calibrated, labels=[0, 1]))
    best_idx = int(np.argmin(losses))
    return float(candidate_temperatures[best_idx])


def build_calibrated_probability_sets(y_val, p_val, p_test):
    y_val = np.asarray(y_val, dtype=int)
    p_val = clip_probabilities(p_val)
    p_test = clip_probabilities(p_test)
    calibrated = {
        "raw": {
            "validation": p_val,
            "test": p_test,
            "extra": {"Calibration_Param": ""},
        }
    }
    if len(np.unique(y_val)) < 2:
        return calibrated

    temperature = fit_temperature_from_validation(y_val, p_val)
    calibrated["temperature"] = {
        "validation": sigmoid_np(probability_to_logit(p_val) / temperature),
        "test": sigmoid_np(probability_to_logit(p_test) / temperature),
        "extra": {"Calibration_Param": f"T={temperature:.4f}"},
    }

    platt = LogisticRegression(solver="lbfgs", max_iter=1000)
    platt.fit(probability_to_logit(p_val).reshape(-1, 1), y_val)
    calibrated["platt"] = {
        "validation": platt.predict_proba(probability_to_logit(p_val).reshape(-1, 1))[:, 1],
        "test": platt.predict_proba(probability_to_logit(p_test).reshape(-1, 1))[:, 1],
        "extra": {
            "Calibration_Param": (
                f"coef={float(platt.coef_[0][0]):.4f};"
                f"intercept={float(platt.intercept_[0]):.4f}"
            )
        },
    }

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(p_val, y_val)
    calibrated["isotonic"] = {
        "validation": isotonic.transform(p_val),
        "test": isotonic.transform(p_test),
        "extra": {"Calibration_Param": "isotonic_validation_fit"},
    }
    return calibrated


def analyze_thresholds_and_calibration(validation_predictions_file, test_predictions_file, output_dir, seed):
    validation_df = pd.read_csv(validation_predictions_file)
    test_df = pd.read_csv(test_predictions_file)
    y_val = validation_df["y_true"].to_numpy(dtype=int)
    y_test = test_df["y_true"].to_numpy(dtype=int)
    p_val = validation_df["prob_adventitious"].to_numpy(dtype=np.float64)
    p_test = test_df["prob_adventitious"].to_numpy(dtype=np.float64)

    calibrated = build_calibrated_probability_sets(y_val, p_val, p_test)
    rows = []
    validation_enriched = validation_df.copy()
    test_enriched = test_df.copy()
    for calibration_name, payload in progress(calibrated.items(), desc=f"Calibration seed {seed}", unit="mode", leave=False):
        val_scores = clip_probabilities(payload["validation"])
        test_scores = clip_probabilities(payload["test"])
        validation_enriched[f"prob_{calibration_name}"] = val_scores
        test_enriched[f"prob_{calibration_name}"] = test_scores
        thresholds = select_thresholds_from_validation(y_val, val_scores)
        for threshold_name, threshold in thresholds.items():
            extra = payload.get("extra", {})
            rows.append(
                binary_metrics_for_threshold(
                    y_val,
                    val_scores,
                    threshold,
                    "validation",
                    calibration_name,
                    threshold_name,
                    seed=seed,
                    extra=extra,
                )
            )
            rows.append(
                binary_metrics_for_threshold(
                    y_test,
                    test_scores,
                    threshold,
                    "test",
                    calibration_name,
                    threshold_name,
                    seed=seed,
                    extra=extra,
                )
            )
    os.makedirs(output_dir, exist_ok=True)
    metrics_file = os.path.join(output_dir, "threshold_calibration_metrics.csv")
    validation_calibrated_file = os.path.join(output_dir, "predicciones_validation_calibradas.csv")
    test_calibrated_file = os.path.join(output_dir, "predicciones_test_calibradas.csv")
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(metrics_file, index=False)
    validation_enriched.to_csv(validation_calibrated_file, index=False)
    test_enriched.to_csv(test_calibrated_file, index=False)
    print(f">>> Threshold/calibration metrics saved to: {metrics_file}")
    return df_metrics


def build_optimizer_and_scheduler(model, num_training_steps):
    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (head_params if name.startswith(("classifier", "projector")) else backbone_params).append(param)
    optimizer = AdamW(
        [{"params": backbone_params, "lr": LEARNING_RATE}, {"params": head_params, "lr": HEAD_LEARNING_RATE}],
        weight_decay=WEIGHT_DECAY,
    )
    warmup_steps = int(round(num_training_steps * WARMUP_RATIO))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler


def save_training_status(seed_dirs, payload):
    status_path = os.path.join(seed_dirs["history"], "training_status.json")
    write_json(status_path, payload)
    return status_path


def train_single_seed(seed, feature_extractor):
    set_seed(seed)
    seed_dirs = build_seed_output_dirs(seed)
    train_dataset = AudioSegmentDataset("train", TRAIN_SEGMENTED_WAV_DIR)
    validation_dataset = AudioSegmentDataset("validation", VALIDATION_SEGMENTED_WAV_DIR)
    test_dataset = AudioSegmentDataset("test", TEST_SEGMENTED_WAV_DIR)

    class_weights, class_weight_summary = resolve_class_weights(train_dataset.labels)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=2,
        problem_type="single_label_classification",
        label2id={"healthy": 0, "adventitious": 1},
        id2label={0: "Healthy", 1: "Adventitious"},
        local_files_only=LOCAL_FILES_ONLY,
        use_safetensors=USE_SAFETENSORS,
        ignore_mismatched_sizes=True,
    )
    if ENABLE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
    if FREEZE_FEATURE_ENCODER:
        if hasattr(model, "freeze_feature_encoder"):
            model.freeze_feature_encoder()
        elif hasattr(model, "freeze_feature_extractor"):
            model.freeze_feature_extractor()
    model.to(torch.device(DEVICE_NAME))
    min_input_samples = compute_min_input_samples(model)
    pass_attention_mask_to_model = use_attention_mask_for_model(model)
    collator = AudioBatchCollator(feature_extractor, min_input_samples)
    train_loader = build_dataloader(train_dataset, collator, TRAIN_BATCH_SIZE, shuffle=True)
    validation_loader = build_dataloader(validation_dataset, collator, EVAL_BATCH_SIZE, shuffle=False)
    test_loader = build_dataloader(test_dataset, collator, EVAL_BATCH_SIZE, shuffle=False)

    dataset_sizes = {
        "train": len(train_dataset),
        "validation": len(validation_dataset),
        "test": len(test_dataset),
    }
    write_seed_manifest(
        seed=seed,
        seed_dirs=seed_dirs,
        dataset_sizes=dataset_sizes,
        training_summary={
            "class_weight_resolved": class_weight_summary,
            "min_input_samples": min_input_samples,
            "model_uses_attention_mask": pass_attention_mask_to_model,
        },
    )

    print(
        f"\n--- wav2vec2 fine-tuning | Seed {seed} ---\n"
        f"train={len(train_dataset)} | validation={len(validation_dataset)} | test={len(test_dataset)}"
    )

    steps_per_epoch = math.ceil(len(train_loader) / max(1, GRAD_ACCUM_STEPS))
    total_training_steps = max(1, NUM_EPOCHS * steps_per_epoch)
    optimizer, scheduler = build_optimizer_and_scheduler(model, total_training_steps)

    best_state_dict = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_validation_metrics = None
    best_key = (-np.inf, -np.inf, -np.inf)
    epochs_without_improvement = 0
    history_rows = []
    history_path = os.path.join(seed_dirs["history"], "training_history.csv")

    save_training_status(
        seed_dirs,
        {
            "seed": seed,
            "status": "running",
            "current_epoch": 0,
            "total_epochs": NUM_EPOCHS,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "history_path": history_path,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    for epoch_idx in range(1, NUM_EPOCHS + 1):
        print(f">>> Starting epoch {epoch_idx}/{NUM_EPOCHS} for seed {seed}...")
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            class_weights=class_weights,
            pass_attention_mask_to_model=pass_attention_mask_to_model,
            seed=seed,
            epoch_idx=epoch_idx,
        )
        validation_metrics = evaluate_model(
            model=model,
            dataloader=validation_loader,
            split_name="validation",
            class_weights=class_weights,
            pass_attention_mask_to_model=pass_attention_mask_to_model,
            seed=seed,
        )
        current_key = evaluation_key(validation_metrics)
        improved = current_key > best_key
        if improved:
            best_key = current_key
            best_epoch = epoch_idx
            best_state_dict = copy.deepcopy(model.state_dict())
            best_validation_metrics = validation_metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history_row = {
            "epoch": epoch_idx,
            "seed": seed,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_auc_roc": validation_metrics["auc_roc"],
            "validation_auc_pr": validation_metrics["auc_pr"],
            "validation_f1_adv": validation_metrics["f1_adv"],
            "validation_recall_adv": validation_metrics["recall_adv"],
            "validation_precision_adv": validation_metrics["precision_adv"],
            "is_best": int(improved),
        }
        history_rows.append(history_row)
        history_df = pd.DataFrame(history_rows)
        history_df.to_csv(history_path, index=False)
        val_auc_display = (
            f"{validation_metrics['auc_roc']:.4f}"
            if np.isfinite(validation_metrics["auc_roc"])
            else "nan"
        )
        status_path = save_training_status(
            seed_dirs,
            {
                "seed": seed,
                "status": "running",
                "current_epoch": epoch_idx,
                "total_epochs": NUM_EPOCHS,
                "best_epoch": best_epoch,
                "best_validation_auc_roc": None if not np.isfinite(best_key[0]) else best_key[0],
                "best_validation_auc_pr": None if not np.isfinite(best_key[1]) else best_key[1],
                "epochs_without_improvement": epochs_without_improvement,
                "latest_metrics": history_row,
                "history_path": history_path,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        print(
            f"Epoch {epoch_idx:02d}/{NUM_EPOCHS} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={validation_metrics['loss']:.4f} | "
            f"val_auc={val_auc_display} | "
            f"best_epoch={best_epoch} | "
            f"history={history_path} | "
            f"status={status_path}"
        )

        if DEVICE_NAME == "mps":
            torch.mps.empty_cache()
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f">>> Early stopping at epoch {epoch_idx}; best epoch was {best_epoch}.")
            break

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(history_path, index=False)
    print(f">>> Training history saved to: {history_path}")
    save_training_status(
        seed_dirs,
        {
            "seed": seed,
            "status": "completed_training_loop",
            "current_epoch": int(history_rows[-1]["epoch"]) if history_rows else 0,
            "total_epochs": NUM_EPOCHS,
            "best_epoch": best_epoch,
            "best_validation_auc_roc": None if not np.isfinite(best_key[0]) else best_key[0],
            "best_validation_auc_pr": None if not np.isfinite(best_key[1]) else best_key[1],
            "epochs_without_improvement": epochs_without_improvement,
            "history_path": history_path,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    model.load_state_dict(best_state_dict)
    best_checkpoint_dir = os.path.join(seed_dirs["checkpoints"], "best_model")
    model.save_pretrained(best_checkpoint_dir, safe_serialization=True)
    feature_extractor.save_pretrained(best_checkpoint_dir)
    checkpoint_summary_path = os.path.join(seed_dirs["checkpoints"], "best_checkpoint_summary.json")
    write_json(
        checkpoint_summary_path,
        {
            "best_epoch": best_epoch,
            "best_validation_selection_key": {
                "auc_roc": best_key[0],
                "auc_pr": best_key[1],
                "neg_loss": best_key[2],
            },
            "class_weight_resolved": class_weight_summary,
            "min_input_samples": min_input_samples,
            "model_uses_attention_mask": pass_attention_mask_to_model,
        },
    )
    print(f">>> Best fine-tuned checkpoint saved to: {best_checkpoint_dir}")

    validation_metrics = evaluate_model(
        model=model,
        dataloader=validation_loader,
        split_name="validation",
        class_weights=class_weights,
        pass_attention_mask_to_model=pass_attention_mask_to_model,
        seed=seed,
    )
    test_metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        split_name="test",
        class_weights=class_weights,
        pass_attention_mask_to_model=pass_attention_mask_to_model,
        seed=seed,
    )
    if DEVICE_NAME == "mps":
        torch.mps.empty_cache()

    report_threshold_strategy, report_threshold, report_thresholds = resolve_report_threshold(
        validation_metrics["y_true"],
        validation_metrics["probabilities"],
    )
    print(
        f">>> Report threshold selected from validation: "
        f"{report_threshold_strategy} = {report_threshold:.6f}"
    )
    validation_metrics = apply_decision_threshold(
        validation_metrics,
        threshold=report_threshold,
        threshold_strategy=report_threshold_strategy,
    )
    test_metrics = apply_decision_threshold(
        test_metrics,
        threshold=report_threshold,
        threshold_strategy=report_threshold_strategy,
    )

    excel_path = os.path.join(seed_dirs["reports"], "metricas.xlsx")
    validation_predictions_path = os.path.join(seed_dirs["predictions"], "predicciones_validation.csv")
    test_predictions_path = os.path.join(seed_dirs["predictions"], "predicciones_test.csv")
    validation_figure_path = os.path.join(seed_dirs["figures"], "analisis_validation.png")
    test_figure_path = os.path.join(seed_dirs["figures"], "analisis_test.png")
    final_rows = [
        export_split_report(
            split_name="validation",
            split_metrics=validation_metrics,
            metadata=validation_metrics["metadata"],
            excel_file=excel_path,
            predictions_file=validation_predictions_path,
            figure_file=validation_figure_path,
            seed=seed,
        ),
        export_split_report(
            split_name="test",
            split_metrics=test_metrics,
            metadata=test_metrics["metadata"],
            excel_file=excel_path,
            predictions_file=test_predictions_path,
            figure_file=test_figure_path,
            seed=seed,
        ),
    ]
    calibration_df = analyze_thresholds_and_calibration(
        validation_predictions_file=validation_predictions_path,
        test_predictions_file=test_predictions_path,
        output_dir=seed_dirs["calibration"],
        seed=seed,
    )
    save_training_status(
        seed_dirs,
        {
            "seed": seed,
            "status": "completed",
            "current_epoch": int(history_rows[-1]["epoch"]) if history_rows else 0,
            "total_epochs": NUM_EPOCHS,
            "best_epoch": best_epoch,
            "best_validation_auc_roc": None if not np.isfinite(best_key[0]) else best_key[0],
            "best_validation_auc_pr": None if not np.isfinite(best_key[1]) else best_key[1],
            "report_threshold_strategy": report_threshold_strategy,
            "report_threshold": report_threshold,
            "available_validation_thresholds": report_thresholds,
            "history_path": history_path,
            "checkpoint_dir": best_checkpoint_dir,
            "predictions_validation": validation_predictions_path,
            "predictions_test": test_predictions_path,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    return final_rows, calibration_df


def run_training():
    assert_segmented_wavs_ready()
    os.makedirs(SUMMARY_OUTPUT_DIR, exist_ok=True)
    write_run_manifest()
    print(f">>> Experiment root: {MODEL_OUTPUT_DIR}")
    print(f">>> Multi-seed summaries: {SUMMARY_OUTPUT_DIR}")
    print(f">>> Training seeds: {SEEDS}")
    if len(SEEDS) == 1:
        print(">>> Screening mode: one seed by default. Run additional seeds only for shortlisted paper-worthy settings.")
    print(f">>> Report threshold strategy: {REPORT_THRESHOLD_STRATEGY}")
    print(f">>> Train segmented dir: {TRAIN_SEGMENTED_WAV_DIR}")
    print(f">>> Validation segmented dir: {VALIDATION_SEGMENTED_WAV_DIR}")
    print(f">>> Test segmented dir: {TEST_SEGMENTED_WAV_DIR}")
    print(f">>> wav2vec2 model: {MODEL_ID}")
    print(f">>> Device: {DEVICE_NAME}")

    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, local_files_only=LOCAL_FILES_ONLY)
    feature_extractor_sampling_rate = int(getattr(feature_extractor, "sampling_rate", SAMPLE_RATE))
    if feature_extractor_sampling_rate != SAMPLE_RATE:
        raise ValueError(
            f"Inconsistent sampling rate: extractor={feature_extractor_sampling_rate}, configured={SAMPLE_RATE}"
        )

    all_results = []
    all_calibration_rows = []
    for seed in progress(SEEDS, desc="Seeds wav2vec2 fine-tune", unit="seed"):
        seed_results, calibration_df = train_single_seed(seed, feature_extractor)
        all_results.extend(seed_results)
        all_calibration_rows.append(calibration_df)

    df_results = pd.DataFrame(all_results)
    resumen_path = os.path.join(SUMMARY_OUTPUT_DIR, "resumen_seeds.csv")
    df_results.to_csv(resumen_path, index=False)
    print(f">>> Per-seed summary saved to: {resumen_path}")

    if all_calibration_rows:
        df_thresholds = pd.concat(all_calibration_rows, ignore_index=True)
        thresholds_resumen_path = os.path.join(SUMMARY_OUTPUT_DIR, "threshold_calibration_resumen_seeds.csv")
        df_thresholds.to_csv(thresholds_resumen_path, index=False)
        print(f">>> Threshold/calibration seed summary saved to: {thresholds_resumen_path}")
    else:
        df_thresholds = None

    if len(SEEDS) > 1:
        metric_cols = ["AUC_ROC", "AUC_PR", "F1-Score_Adv", "Recall_Adv", "Precision_Adv", "TN", "FP", "FN", "TP"]
        media_std_path = os.path.join(SUMMARY_OUTPUT_DIR, "media_std_por_split.csv")
        df_results.groupby("Split")[metric_cols].agg(["mean", "std"]).to_csv(media_std_path)
        print(f">>> Mean/std by split saved to: {media_std_path}")
        if df_thresholds is not None:
            threshold_metric_cols = ["Accuracy", "AUC_ROC", "AUC_PR", "Brier", "LogLoss", "F1_Adv", "Recall_Adv", "Specificity", "Precision_Adv"]
            thresholds_media_std_path = os.path.join(SUMMARY_OUTPUT_DIR, "threshold_calibration_media_std.csv")
            (
                df_thresholds.groupby(["Calibration", "Threshold_Strategy", "Split"])[threshold_metric_cols]
                .agg(["mean", "std"])
                .to_csv(thresholds_media_std_path)
            )
            print(f">>> Threshold/calibration mean/std saved to: {thresholds_media_std_path}")


def main():
    run_training()


if __name__ == "__main__":
    main()
