from datetime import datetime
import os
import json
import shutil

# El script localiza la raíz del proyecto para que todas las rutas operativas
# sean relativas a Fix_Paper1, independientemente de la subcarpeta donde viva.
_SCRIPT_DIR = os.path.normpath(os.path.dirname(__file__) or ".")


def _find_project_root(start_dir):
    current_dir = os.path.abspath(start_dir)
    while True:
        if os.path.exists(os.path.join(current_dir, "environment.yml")) and os.path.isdir(os.path.join(current_dir, "ficheros")):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise FileNotFoundError("No se pudo localizar la raíz de Fix_Paper1.")
        current_dir = parent_dir


_PROJECT_ROOT = _find_project_root(_SCRIPT_DIR)
os.chdir(_PROJECT_ROOT)
_PROJECT_DIR = "."
_CACHE_DIR = os.path.join(_PROJECT_DIR, ".cache")
_MPLCONFIGDIR = os.path.join(_PROJECT_DIR, ".matplotlib")
_DEFAULT_CPU_THREAD_COUNT = str(min(8, os.cpu_count() or 4))
os.environ.setdefault("XDG_CACHE_HOME", _CACHE_DIR)
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(_CACHE_DIR, "numba"))
os.environ.setdefault("MPLCONFIGDIR", _MPLCONFIGDIR)
os.environ.setdefault("OMP_NUM_THREADS", _DEFAULT_CPU_THREAD_COUNT)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", os.environ["OMP_NUM_THREADS"])
os.environ.setdefault("NUMEXPR_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
_CPU_THREAD_COUNT = os.environ["OMP_NUM_THREADS"]
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)


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
        raise ValueError(f"{env_name} no contiene ninguna seed valida.")
    return values


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

import librosa
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from collections import Counter, defaultdict
from scipy.signal import butter, lfilter
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import seaborn as sns

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

#---------------------------------------------------------------------------------------------------------------
# Parámetros de frecuencia de muestreo final y parámetros del filtrado BP
#---------------------------------------------------------------------------------------------------------------
SR_TARGET = 8000
LOWCUT = 50.0
HIGHCUT = 3999.0

#---------------------------------------------------------------------------------------------------------------
# Parámetros para el espectrograma
#---------------------------------------------------------------------------------------------------------------
PADDING_MODE = 0  # 1 para PADDING (Usa percentil), 0 para RESIZE DINÁMICO (Usa SIZE)
ETIQUETA_PREPROC="Preprocesamiento con resize-dinámico, sin padding ni data-augmentation"
N_MELS = 64
SIZE = 64 # Si PADDING_MODE = 0, se reescala el espectrigrama a este tamaño, para subsanar las diferentes longitudes de los audios
PERCENTIL = 90
DURACION_PERCENTIL = 0.90

#---------------------------------------------------------------------------------------------------------------
# Rutas
#---------------------------------------------------------------------------------------------------------------
DATASET_NAME = "0 SPRSound"
SPLIT_SEED = int(os.environ.get("SPLIT_SEED", 42))  # R1: configurable para particiones alternativas
VALIDATION_FRACTION = 0.20
SEEDS = parse_int_list_env("EXPERIMENT_SEEDS", [42, 43, 44])
SEED_LIST_LABEL = "-".join(str(seed) for seed in SEEDS)
TRAINING_SEEDS_TAG = f"seeds_{SEED_LIST_LABEL}"
EXCLUDE_RECORD_ANNOTATIONS = {"Poor Quality"}
EXCLUDE_MISSING_PATIENT_ID = True

# Metadatos de experimento SPRSound.
DATASET_LABEL = os.environ.get("DATASET_LABEL", "sprsound")
TRAIN_DATASET_LABEL = os.environ.get("TRAIN_DATASET_LABEL", DATASET_LABEL)
VALIDATION_DATASET_LABEL = os.environ.get("VALIDATION_DATASET_LABEL", TRAIN_DATASET_LABEL)
TEST_DATASET_LABEL = os.environ.get("TEST_DATASET_LABEL", DATASET_LABEL)
DATASET_PROTOCOL = os.environ.get(
    "DATASET_PROTOCOL",
    f"{slugify(TRAIN_DATASET_LABEL)}_to_{slugify(TEST_DATASET_LABEL)}",
)
EXPERIMENT_BLOCK = os.environ.get("EXPERIMENT_BLOCK", "00_cnn_scratch_baseline")
MODEL_FAMILY = os.environ.get("MODEL_FAMILY", "cnn")
MODEL_DISPLAY_NAME = os.environ.get("MODEL_DISPLAY_NAME", "ResNeXt-Attention")
BACKBONE_NAME = os.environ.get("BACKBONE_NAME", "biocas_resnext_attention")
TRAINING_MODE = os.environ.get("TRAINING_MODE", "scratch")
CLASSIFIER_TYPE = os.environ.get("CLASSIFIER_TYPE", "trainable_mlp")
EMBEDDING_MODE = os.environ.get("EMBEDDING_MODE", "none")
FINETUNING_SCOPE = os.environ.get("FINETUNING_SCOPE", "none")
EXPERIMENT_NOTES = os.environ.get("EXPERIMENT_NOTES", "")

PLANNED_EXPERIMENT_STRUCTURE = {
    "00_cnn_scratch_baseline": {
        "purpose": "CNN entrenada desde cero; baseline para comparar cnn scratch vs pretrained.",
        "models": ["cnn"],
        "dataset_protocols": ["sprsound_to_sprsound"],
    }
}

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

# Ruta de los ficheros segmentados por evento. Se separa por dataset y seed para dejar claro el protocolo usado.
OUTPUT_BASE_DIR = os.path.join(_PROJECT_DIR, "ficheros", "1 Segmentados_WAV", DATASET_NAME, f"seed_{SPLIT_SEED}")
CARPETAS_A_REVISAR = ["segmentado_train", "segmentado_validation", "segmentado_test"]
if PADDING_MODE:
    DEFAULT_PATH_SALIDA_NPY = os.path.join(_PROJECT_DIR, f"ficheros/2 Espectrogramas/{DATASET_NAME}_seed{SPLIT_SEED}_padding_MEL{N_MELS}_PERCENTIL{PERCENTIL}")
else:
    DEFAULT_PATH_SALIDA_NPY = os.path.join(_PROJECT_DIR, f"ficheros/2 Espectrogramas/{DATASET_NAME}_seed{SPLIT_SEED}_resize_dinamico_MEL{N_MELS}_SIZE{SIZE}")

PATH_SALIDA_NPY = os.environ.get("PATH_SALIDA_NPY", DEFAULT_PATH_SALIDA_NPY)
TRAIN_PATH = os.environ.get("TRAIN_NPY_PATH", os.path.join(PATH_SALIDA_NPY, "npy", "segmentado_train"))
VALIDATION_PATH = os.environ.get("VALIDATION_NPY_PATH", os.path.join(PATH_SALIDA_NPY, "npy", "segmentado_validation"))
TEST_PATH = os.environ.get("TEST_NPY_PATH", os.path.join(PATH_SALIDA_NPY, "npy", "segmentado_test"))

#---------------------------------------------------------------------------------------------------------------
# Parámetros entrenamiento modelos ML
#---------------------------------------------------------------------------------------------------------------
WEIGHT1 = 1.0
WEIGHT2 = 3.5  #  mejor resultado con epoch=50, drop=.2 y w2=3.5
DROPOUT = 0.2
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0001
GRUPOS = 4
CHECKPOINT_SELECTION_METRIC = "pr_auc"  # Opciones: "f1", "roc_auc", "pr_auc"
DATALOADER_NUM_WORKERS = int(os.environ.get("DATALOADER_NUM_WORKERS", min(4, max(0, (os.cpu_count() or 2) // 2))))
DATALOADER_PREFETCH_FACTOR = 2

#---------------------------------------------------------------------------------------------------------------
# Rutas informes de salida y modelo entrenado
#---------------------------------------------------------------------------------------------------------------
FECHA_EJECUCION = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ID = os.environ.get("RUN_ID", f"run_{FECHA_EJECUCION}")
PREPROCESSING_TAG = (
    f"padding_mel{N_MELS}_p{PERCENTIL}"
    if PADDING_MODE
    else f"resize_mel{N_MELS}_size{SIZE}"
)
MODEL_SETUP_TAG = "_".join(
    slugify(part)
    for part in (MODEL_FAMILY, BACKBONE_NAME, TRAINING_MODE, CLASSIFIER_TYPE, FINETUNING_SCOPE)
    if str(part).strip() and str(part).strip().lower() != "none"
)
MODEL_SETUP_TAG = MODEL_SETUP_TAG or slugify(MODEL_FAMILY)
RUN_BASE_NAME = (
    f"{slugify(MODEL_FAMILY)}_{slugify(TRAINING_MODE)}_{slugify(CLASSIFIER_TYPE)}_"
    f"splitSeed{SPLIT_SEED}_{TRAINING_SEEDS_TAG}_ckpt_{CHECKPOINT_SELECTION_METRIC}_"
    f"{PREPROCESSING_TAG}_{RUN_ID}"
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


def build_seed_output_dirs(seed, create=True):
    seed_root = os.path.join(MODEL_OUTPUT_DIR, f"seed_{seed}")
    dirs = {
        "root": seed_root,
        "checkpoints": os.path.join(seed_root, "checkpoints"),
        "reports": os.path.join(seed_root, "reports"),
        "predictions": os.path.join(seed_root, "predictions"),
        "figures": os.path.join(seed_root, "figures"),
        "calibration": os.path.join(seed_root, "calibration"),
    }
    if create:
        for path in dirs.values():
            os.makedirs(path, exist_ok=True)
    return dirs


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_run_manifest(target_width=None):
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_dir": _PROJECT_DIR,
        "current_experiment": EXPERIMENT_METADATA,
        "planned_experiment_structure": PLANNED_EXPERIMENT_STRUCTURE,
        "notes": EXPERIMENT_NOTES,
        "paths": {
            "model_output_dir": MODEL_OUTPUT_DIR,
            "summary_output_dir": SUMMARY_OUTPUT_DIR,
            "input_base_dir": INPUT_BASE_DIR,
            "split_manifest_json": SPLIT_MANIFEST_JSON,
            "split_manifest_csv": SPLIT_MANIFEST_CSV,
            "split_manifests_by_split": {
                "train": {
                    "json": TRAIN_SPLIT_MANIFEST_JSON,
                    "csv": TRAIN_SPLIT_MANIFEST_CSV,
                },
                "validation": {
                    "json": VALIDATION_SPLIT_MANIFEST_JSON,
                    "csv": VALIDATION_SPLIT_MANIFEST_CSV,
                },
                "test": {
                    "json": TEST_SPLIT_MANIFEST_JSON,
                    "csv": TEST_SPLIT_MANIFEST_CSV,
                },
            },
            "segmented_wav_dir": OUTPUT_BASE_DIR,
            "spectrogram_dir": PATH_SALIDA_NPY,
            "default_spectrogram_dir": DEFAULT_PATH_SALIDA_NPY,
            "train_path": TRAIN_PATH,
            "validation_path": VALIDATION_PATH,
            "test_path": TEST_PATH,
        },
        "split_config": {
            "split_seed": SPLIT_SEED,
            "validation_fraction": VALIDATION_FRACTION,
            "training_seeds": SEEDS,
            "exclude_missing_patient_id": EXCLUDE_MISSING_PATIENT_ID,
        },
        "preprocessing": {
            "sr_target": SR_TARGET,
            "lowcut": LOWCUT,
            "highcut": HIGHCUT,
            "padding_mode": PADDING_MODE,
            "n_mels": N_MELS,
            "size": SIZE,
            "percentil": PERCENTIL,
            "duracion_percentil": DURACION_PERCENTIL,
            "target_width": target_width,
            "label": ETIQUETA_PREPROC,
        },
        "training": {
            "weight1": WEIGHT1,
            "weight2": WEIGHT2,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "groups": GRUPOS,
            "checkpoint_selection_metric": CHECKPOINT_SELECTION_METRIC,
            "dataloader_num_workers": DATALOADER_NUM_WORKERS,
            "dataloader_prefetch_factor": DATALOADER_PREFETCH_FACTOR,
        },
    }


def write_run_manifest(target_width=None):
    manifest_path = os.path.join(MODEL_OUTPUT_DIR, "run_manifest.json")
    write_json(manifest_path, build_run_manifest(target_width=target_width))
    print(f">>> Manifiesto del experimento guardado en: {manifest_path}")


def write_seed_manifest(seed, seed_dirs, dataset_sizes):
    manifest = build_run_manifest()
    manifest.update({
        "seed": seed,
        "seed_output_dirs": seed_dirs,
        "dataset_sizes": dataset_sizes,
    })
    manifest_path = os.path.join(seed_dirs["root"], "seed_manifest.json")
    write_json(manifest_path, manifest)
    print(f">>> Manifiesto seed {seed} guardado en: {manifest_path}")


def validate_implemented_experiment():
    implemented = (
        slugify(MODEL_FAMILY) == "cnn"
        and slugify(TRAINING_MODE) == "scratch"
        and slugify(CLASSIFIER_TYPE) == "trainable-mlp"
    )
    if implemented:
        return

    raise NotImplementedError(
        "Este script aun solo implementa la CNN entrenada desde cero con clasificador MLP. "
        "La estructura de resultados ya contempla frozen embeddings/logistic regression, "
        "YAMNet, wav2vec2 y fine-tuning, pero esos backends deben implementarse antes de "
        "ejecutarlos para no generar resultados mal etiquetados."
    )

# Semilla y GPU
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f">>> Semilla establecida en: {seed}")


def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


set_seed(42)
if hasattr(torch, "set_num_threads"):
    torch.set_num_threads(int(_CPU_THREAD_COUNT))
device = get_best_device()
PIN_MEMORY = device.type == "cuda"
NON_BLOCKING_TRANSFER = device.type == "cuda"
print(f">>> Dispositivo seleccionado: {device}")
print(f">>> DataLoader workers: {DATALOADER_NUM_WORKERS} | threads CPU: {_CPU_THREAD_COUNT}")


def normalize_event_label(label):
    """Normaliza las etiquetas SPRSound para que coincidan con los nombres de carpetas."""
    label = str(label).strip().replace("+", "_").replace(" ", "_")
    return "_".join(part for part in label.split("_") if part)


EVENT_TO_BINARY = {
    'Normal': 0,
    'Crackle': 1,
    'Rhonchi': 1,
    'Wheeze': 1,
    'Stridor': 1,
    'Coarse_Crackle': 1,
    'Fine_Crackle': 1,
    'Wheeze_Crackle': 1
}


def parse_sprsound_filename(filename):
    # Formato esperado: patient_age_gender_location_recording.wav/json
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.split("_")
    metadata = {
        "recording_id": stem,
        "patient_id": "",
        "age": "",
        "gender": "",
        "location": "",
        "recording_number": "",
        "filename_format_ok": len(parts) == 5,
        "patient_id_missing": True,
    }
    if len(parts) == 5:
        patient_id, age, gender, location, recording_number = parts
        metadata.update({
            "patient_id": patient_id,
            "age": age,
            "gender": gender,
            "location": location,
            "recording_number": recording_number,
            "patient_id_missing": patient_id == "",
        })
    else:
        metadata["recording_number"] = parts[-1] if parts else stem

    if metadata["patient_id_missing"]:
        # Si el paciente no está en el nombre, no se puede verificar independencia por paciente.
        # Se usa un grupo único por recording para no mezclar ficheros con el mismo stem.
        metadata["patient_group"] = f"MISSING_PATIENT::{stem}"
    else:
        metadata["patient_group"] = metadata["patient_id"]

    return metadata


def load_annotation(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("event_annotation", [])
    return data


def scan_sprsound_recordings(json_dir, wav_dir, source_split):
    records = []
    if not os.path.exists(json_dir):
        raise FileNotFoundError(f"No existe la carpeta JSON: {json_dir}")
    if not os.path.exists(wav_dir):
        raise FileNotFoundError(f"No existe la carpeta WAV: {wav_dir}")

    json_files = sorted(f for f in os.listdir(json_dir) if f.endswith(".json"))
    for json_file in progress(json_files, desc=f"Escaneando {source_split}", unit="json", leave=False):
        metadata = parse_sprsound_filename(json_file)
        wav_path = os.path.join(wav_dir, metadata["recording_id"] + ".wav")
        annotation = load_annotation(os.path.join(json_dir, json_file))
        event_labels = [normalize_event_label(ev.get("type", "")) for ev in annotation.get("event_annotation", [])]
        record_annotation = annotation.get("record_annotation")

        records.append({
            **metadata,
            "source_split": source_split,
            "json_path": os.path.join(json_dir, json_file),
            "wav_path": wav_path,
            "wav_exists": os.path.exists(wav_path),
            "record_annotation": record_annotation,
            "n_events": len(event_labels),
            "n_normal_events": sum(1 for label in event_labels if label == "Normal"),
            "n_adventitious_events": sum(1 for label in event_labels if EVENT_TO_BINARY.get(label) == 1),
            "event_types": sorted(set(event_labels)),
            "event_type_counts": dict(Counter(event_labels)),
        })

    return records


def split_train_validation_by_patient(records, validation_fraction=0.20, seed=42):
    patient_groups = defaultdict(list)
    for record in records:
        if record["record_annotation"] in EXCLUDE_RECORD_ANNOTATIONS:
            continue
        if not record["wav_exists"]:
            continue
        if EXCLUDE_MISSING_PATIENT_ID and record["patient_id_missing"]:
            continue
        patient_groups[record["patient_group"]].append(record)

    rng = random.Random(seed)
    positive_groups, negative_groups = [], []
    for group_records in patient_groups.values():
        has_adventitious = any(r["n_adventitious_events"] > 0 for r in group_records)
        if has_adventitious:
            positive_groups.append(group_records)
        else:
            negative_groups.append(group_records)

    validation_groups = []
    for bucket in (positive_groups, negative_groups):
        rng.shuffle(bucket)
        bucket_record_count = sum(len(group_records) for group_records in bucket)
        target_count = max(1, int(round(bucket_record_count * validation_fraction))) if bucket_record_count else 0
        selected_count = 0
        for group_records in bucket:
            if selected_count >= target_count and validation_groups:
                break
            validation_groups.append(group_records)
            selected_count += len(group_records)

    validation_patient_groups = {group_records[0]["patient_group"] for group_records in validation_groups}
    train_records, validation_records = [], []
    for patient_group, group_records in patient_groups.items():
        target = validation_records if patient_group in validation_patient_groups else train_records
        target.extend(group_records)

    return sorted(train_records, key=lambda r: r["recording_id"]), sorted(validation_records, key=lambda r: r["recording_id"])


def summarize_records(records):
    patient_ids = {r["patient_id"] for r in records if r["patient_id"]}
    return {
        "recordings": len(records),
        "patients_with_id": len(patient_ids),
        "recordings_missing_patient_id": sum(1 for r in records if r["patient_id_missing"]),
        "record_annotations": dict(Counter(str(r["record_annotation"]) for r in records)),
        "event_type_counts": dict(sum((Counter(r["event_type_counts"]) for r in records), Counter())),
        "events_total": sum(r["n_events"] for r in records),
        "normal_events": sum(r["n_normal_events"] for r in records),
        "adventitious_events": sum(r["n_adventitious_events"] for r in records),
    }


def assert_no_overlap(splits):
    split_names = list(splits.keys())
    overlaps = {}
    for i, split_a in enumerate(split_names):
        for split_b in split_names[i + 1:]:
            ids_a = {r["patient_id"] for r in splits[split_a] if r["patient_id"]}
            ids_b = {r["patient_id"] for r in splits[split_b] if r["patient_id"]}
            patient_overlap = sorted(ids_a & ids_b)

            rec_a = {r["recording_id"] for r in splits[split_a]}
            rec_b = {r["recording_id"] for r in splits[split_b]}
            recording_overlap = sorted(rec_a & rec_b)

            overlaps[f"{split_a}__{split_b}"] = {
                "patient_id_overlap": patient_overlap,
                "recording_id_overlap": recording_overlap,
            }
            if patient_overlap or recording_overlap:
                raise ValueError(
                    f"Leakage detectado entre {split_a} y {split_b}. "
                    f"Pacientes: {patient_overlap[:10]}, recordings: {recording_overlap[:10]}"
                )
    return overlaps


def write_split_manifest(splits, overlaps):
    os.makedirs(SPLIT_MANIFEST_DIR, exist_ok=True)
    manifest = {
        "dataset": DATASET_NAME,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "split_seed": SPLIT_SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "policy": {
            "test_policy": "SPRSound official test_json/test_wav; never used for checkpoint selection",
            "validation_policy": "patient-independent split sampled from train_json/train_wav",
            "checkpoint_selection": "validation only",
            "excluded_record_annotations": sorted(EXCLUDE_RECORD_ANNOTATIONS),
            "missing_patient_id_policy": "excluded from all splits because patient independence cannot be verified",
        },
        "input_dirs": {
            "train_json": os.path.join(INPUT_BASE_DIR, "train_json"),
            "train_wav": os.path.join(INPUT_BASE_DIR, "train_wav"),
            "test_json": os.path.join(INPUT_BASE_DIR, "test_json"),
            "test_wav": os.path.join(INPUT_BASE_DIR, "test_wav"),
        },
        "overlaps": overlaps,
        "summary": {split: summarize_records(records) for split, records in splits.items()},
        "splits": {
            split: [
                {
                    key: record[key]
                    for key in (
                        "recording_id", "patient_id", "patient_group", "patient_id_missing",
                        "age", "gender", "location", "recording_number", "record_annotation",
                        "n_events", "n_normal_events", "n_adventitious_events", "event_types",
                        "event_type_counts",
                        "wav_exists", "source_split"
                    )
                }
                for record in records
            ]
            for split, records in splits.items()
        },
    }
    with open(SPLIT_MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    rows = []
    for split, records in progress(splits.items(), desc="Escribiendo manifiesto de splits", unit="split", leave=False):
        for record in progress(records, desc=f"Manifest {split}", unit="rec", leave=False):
            rows.append({
                "split": split,
                "recording_id": record["recording_id"],
                "patient_id": record["patient_id"],
                "patient_group": record["patient_group"],
                "patient_id_missing": record["patient_id_missing"],
                "age": record["age"],
                "gender": record["gender"],
                "location": record["location"],
                "recording_number": record["recording_number"],
                "record_annotation": record["record_annotation"],
                "n_events": record["n_events"],
                "n_normal_events": record["n_normal_events"],
                "n_adventitious_events": record["n_adventitious_events"],
                "event_types": "|".join(record["event_types"]),
                "event_type_counts": json.dumps(record["event_type_counts"], ensure_ascii=False, sort_keys=True),
                "json_path": record["json_path"],
                "wav_path": record["wav_path"],
            })
    pd.DataFrame(rows).to_csv(SPLIT_MANIFEST_CSV, index=False)
    print(f">>> Manifiesto de splits guardado en: {SPLIT_MANIFEST_JSON}")
    print(f">>> Tabla de recordings por split guardada en: {SPLIT_MANIFEST_CSV}")


def preparar_splits_sprsound():
    train_source_records = scan_sprsound_recordings(
        os.path.join(INPUT_BASE_DIR, "train_json"),
        os.path.join(INPUT_BASE_DIR, "train_wav"),
        "official_train",
    )
    test_records = scan_sprsound_recordings(
        os.path.join(INPUT_BASE_DIR, "test_json"),
        os.path.join(INPUT_BASE_DIR, "test_wav"),
        "official_test",
    )
    test_source_records = test_records
    test_records = [
        record for record in test_records
        if (
            record["record_annotation"] not in EXCLUDE_RECORD_ANNOTATIONS
            and record["wav_exists"]
            and (not EXCLUDE_MISSING_PATIENT_ID or not record["patient_id_missing"])
        )
    ]

    train_records, validation_records = split_train_validation_by_patient(
        train_source_records,
        validation_fraction=VALIDATION_FRACTION,
        seed=SPLIT_SEED,
    )
    splits = {
        "train": train_records,
        "validation": validation_records,
        "test": sorted(test_records, key=lambda r: r["recording_id"]),
    }
    overlaps = assert_no_overlap(splits)
    excluded_records = []
    for record in train_source_records + test_source_records:
        reasons = []
        if record["record_annotation"] in EXCLUDE_RECORD_ANNOTATIONS:
            reasons.append("excluded_record_annotation")
        if not record["wav_exists"]:
            reasons.append("missing_wav")
        if EXCLUDE_MISSING_PATIENT_ID and record["patient_id_missing"]:
            reasons.append("missing_patient_id")
        if reasons:
            excluded_records.append({
                "recording_id": record["recording_id"],
                "source_split": record["source_split"],
                "record_annotation": record["record_annotation"],
                "patient_id_missing": record["patient_id_missing"],
                "wav_exists": record["wav_exists"],
                "reasons": reasons,
            })
    overlaps["excluded_records"] = excluded_records
    write_split_manifest(splits, overlaps)

    print("\n" + "=" * 40)
    print("RESUMEN DE SPLITS SPRSound")
    print("=" * 40)
    for split, summary in {name: summarize_records(records) for name, records in splits.items()}.items():
        print(
            f"{split}: {summary['recordings']} recordings | "
            f"{summary['patients_with_id']} pacientes con ID | "
            f"{summary['recordings_missing_patient_id']} sin patient_id | "
            f"{summary['normal_events']} eventos normales | "
            f"{summary['adventitious_events']} eventos adventicios"
        )
    return splits


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)


def reset_output_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def extrae_eventos(records, output_base):

    # Hace la segmentación de audios: de un WAV largo + su JSON de anotaciones,
    # crea clips cortos por evento y los organiza por clase. Guarda cada recorte como un .wav
    # en una carpeta por clase: por ejemplo, Normal, Wheeze, Coarse_Crackle, etc.

    reset_output_dir(output_base)
    stats = Counter()

    print(f"\n--- Procesando split segmentado: {os.path.basename(output_base)} ({len(records)} recordings) ---")

    for record in progress(records, desc=f"Segmentando {os.path.basename(output_base)}", unit="rec"):
        try:
            data = load_annotation(record["json_path"])
            base_name = record["recording_id"]
            wav_path = record["wav_path"]

            if not os.path.exists(wav_path):
                continue

            y_full, _ = librosa.load(wav_path, sr=SR_TARGET)

            for i, event in enumerate(data.get('event_annotation', [])):
                start_ms = float(event['start'])
                end_ms = float(event['end'])
                label = normalize_event_label(event.get('type', ''))

                if label not in EVENT_TO_BINARY:
                    print(f"Etiqueta no reconocida en {base_name}: {event.get('type')}")
                    continue

                if (end_ms - start_ms) < 50: continue

                start_sample = int((start_ms / 1000) * SR_TARGET)
                end_sample = int((end_ms / 1000) * SR_TARGET)
                y_event = y_full[start_sample:end_sample]

                if len(y_event) == 0: continue

                class_dir = os.path.join(output_base, label)
                os.makedirs(class_dir, exist_ok=True)

                sf.write(os.path.join(class_dir, f"{base_name}_ev{i}.wav"), y_event, SR_TARGET)
                stats[label] += 1

        except Exception as e:
            print(f"Error en {record.get('recording_id', 'recording_desconocido')}: {e}")

    return stats


def auditar_dataset_biocas(base_path, subsets):

    # cuenta y resume cuántos audios sanos y adventicios hay en cada parte del dataset.

    task_1_dict = {
        label: ('Sano (0)' if binary == 0 else 'Adventicio (1)')
        for label, binary in EVENT_TO_BINARY.items()
    }
    for subset in subsets:
        subset_path = os.path.join(base_path, subset)
        if not os.path.exists(subset_path): continue
        conteo_binario = Counter({'Sano (0)': 0, 'Adventicio (1)': 0})
        for clase in os.listdir(subset_path):
            ruta_clase = os.path.join(subset_path, clase)
            if os.path.isdir(ruta_clase):
                num_files = len([f for f in os.listdir(ruta_clase) if f.endswith('.wav')])
                tipo_binario = task_1_dict.get(normalize_event_label(clase), "Desconocido")
                if tipo_binario in conteo_binario:
                    conteo_binario[tipo_binario] += num_files
        print(f"Subconjunto {subset}: Sano: {conteo_binario['Sano (0)']} | Adv: {conteo_binario['Adventicio (1)']}")


def calcular_ancho_percentil(base_dir, subsets, p=0.90):

    # Calcula un ancho “típico” para los espectrogramas a partir de la duración de los audios.
    # Recorre todos los .wav de los subconjuntos indicados.
    # Obtiene la duración de cada audio.
    # Convierte esa duración en número de frames/columnas del espectrograma.
    # Calcula el percentil que se le pase (p, por defecto 0.90).
    # Devuelve ese valor como ancho objetivo.
    # Ese ancho se usa luego para que todos los espectrogramas tengan un tamaño parecido:
    #       - Si un audio es más corto, se rellena con ceros al final del espectrograma
    #       - Si es más largo, se recorta por la derecha para que tenga la duración marcada por el percentil

    wav_files = []
    for subset in subsets:
        path = os.path.join(base_dir, subset)
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith('.wav'):
                    wav_files.append(os.path.join(root, f))

    duraciones_frames = []
    for wav_file in progress(wav_files, desc="Calculando percentil de duración", unit="wav", leave=False):
        d = librosa.get_duration(path=wav_file, sr=SR_TARGET)
        # El hop_length 256 define cuántas columnas resultan
        frames = int((d * SR_TARGET) / 256) + 1
        duraciones_frames.append(frames)
    if not duraciones_frames:
        raise ValueError(f"No se encontraron WAV para calcular el percentil en {base_dir} / {subsets}")
    ancho_target = int(np.percentile(duraciones_frames, p * 100))
    #print(f"Ancho objetivo detectado: {ancho_target} columnas.")
    return ancho_target


def wav_to_clean_mel(wav_path,target_width):
    y, _ = librosa.load(wav_path, sr=SR_TARGET)
    if len(y) == 0: return None
    y = librosa.util.normalize(y)
    y = apply_filter(y, LOWCUT, HIGHCUT, SR_TARGET)

    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=SR_TARGET, n_mels=N_MELS, fmax=4000, n_fft=1024, hop_length=256
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_norm = np.clip((mel_db + 80) / 80, 0, 1)

    if PADDING_MODE == 0:
        # RESIZE: Aquí sí usamos el valor de SIZE (target_width)
        mel_final = resize(mel_norm, (N_MELS, target_width), anti_aliasing=True, preserve_range=True)
    else:
        # PADDING: Ignoramos SIZE y usamos el percentil (target_width)
        if mel_norm.shape[1] < target_width:
            pad_width = target_width - mel_norm.shape[1]
            mel_final = np.pad(mel_norm, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_final = mel_norm[:, :target_width]

        # Forzar el alto a N_MELS sin tocar el ancho (target_width)
        if mel_final.shape[0] != N_MELS:
            mel_final = resize(mel_final, (N_MELS, target_width), preserve_range=True)

    return mel_final.astype(np.float32)


def procesar_y_guardar(base_dir, output_root, subsets,target_width):

    # convierte los audios .wav segmentados en espectrogramas Mel y los guarda como .npy y .png.
    # Para cada subconjunto y cada clase:
    # 1) Busca todos los .wav dentro de esa carpeta.
    # 2) Carga cada audio.
    # 3) Lo normaliza y le aplica un filtrado de banda.
    # 4) Calcula su Mel espectrograma.
    # 5) Lo ajusta al ancho final usando padding si falta ancho, o recorte si sobra.
    # 6) Guarda el resultado en npy para usarlo luego en entrenamiento y en png para visualizarlo

    for path in (os.path.join(output_root, "npy"), os.path.join(output_root, "visualizacion")):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    for subset in progress(subsets, desc="Generando espectrogramas", unit="split"):
        input_path = os.path.join(base_dir, subset)
        npy_root = os.path.join(output_root, "npy", subset)
        png_root = os.path.join(output_root, "visualizacion", subset)

        print(f"\n>>> Procesando subset: {subset}")

        tareas = []
        for clase in sorted(os.listdir(input_path)):
            clase_in = os.path.join(input_path, clase)
            if not os.path.isdir(clase_in): continue

            os.makedirs(os.path.join(npy_root, clase), exist_ok=True)
            os.makedirs(os.path.join(png_root, clase), exist_ok=True)

            files = sorted(glob.glob(os.path.join(clase_in, "*.wav")))
            tareas.extend((f, clase) for f in files)

        for f, clase in progress(tareas, desc=f"Espectrogramas {subset}", unit="wav", leave=False):
            mel = wav_to_clean_mel(f,target_width)  # Usamos el ancho final calculado
            if mel is not None:
                nombre_base = os.path.basename(f).replace('.wav', '')
                np.save(os.path.join(npy_root, clase, f"{nombre_base}.npy"), mel)
                plt.imsave(os.path.join(png_root, clase, f"{nombre_base}.png"), mel, cmap='magma', origin='lower')

    print(f"\n¡Conversión terminada! Ancho resultante: {target_width}")

def segmentar_carpetas(splits):
    # Es una función que recorre una carpeta de anotaciones JSON y una carpeta de audios WAV, y:
    # 1) lee cada JSON
    # 2) busca el WAV con el mismo nombre
    # 3) extrae cada evento anotado
    # 4) recorta ese fragmento de audio
    # 5) lo guarda como un .wav separado
    # 6) lo organiza en carpetas por clase

    subsets_config = [
        ("train", "segmentado_train"),
        ("validation", "segmentado_validation"),
        ("test", "segmentado_test"),
    ]
    # --- EJECUCIÓN Y RESUMEN ---
    resumen_final = {}

    for split_name, out_rel in subsets_config:
        p_out = os.path.join(OUTPUT_BASE_DIR, out_rel)

        conteo = extrae_eventos(splits[split_name], p_out)
        resumen_final[out_rel] = conteo

    print("\n" + "="*40)
    print("RESUMEN DE EVENTOS EXTRAÍDOS")
    print("="*40)
    for subset, counts in resumen_final.items():
        print(f"\nSubconjunto: {subset}")
        for label, total in counts.items():
            print(f"  - {label}: {total}")

# ==========================================
# ARQUITECTURA BioCAS-ResNeXt-Attention
# ==========================================
class SEBlock(nn.Module):
    """Mecanismo de Atención de Canal (Squeeze-and-Excitation)"""

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BioCASResNextBlock(nn.Module):
    """Bloque Residual con Convolución Agrupada"""

    def __init__(self, in_channels, out_channels, stride=1, groups=GRUPOS):
        super(BioCASResNextBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.se = SEBlock(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv(x)
        out = self.se(out)
        out += self.shortcut(x)
        return F.relu(out)


class BioCAS_Advanced_CNN(nn.Module):
    def __init__(self):
        super(BioCAS_Advanced_CNN, self).__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer1 = BioCASResNextBlock(64, 64, stride=1)
        self.layer2 = BioCASResNextBlock(64, 128, stride=2)
        self.layer3 = BioCASResNextBlock(128, 256, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        return self.classifier(x)


# ==========================================
# DATASET Y CARGA (SIN DATA AUGMENTATION)
# ==========================================
class BioCASNumpyDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        self.class_to_idx = EVENT_TO_BINARY
        unmapped_classes = []

        # Escaneo de carpetas y mapeo binario
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"No existe el directorio NPY: {root_dir}")

        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path): continue

            label = self.class_to_idx.get(normalize_event_label(class_name))
            if label is None:
                npy_count = sum(name.endswith('.npy') for name in os.listdir(class_path))
                if npy_count:
                    unmapped_classes.append((class_name, npy_count))
                continue

            for f in sorted(os.listdir(class_path)):
                if f.endswith('.npy'):
                    self.samples.append((os.path.join(class_path, f), label))

        if unmapped_classes:
            details = ", ".join(f"{name} ({count} .npy)" for name, count in unmapped_classes)
            raise ValueError(
                f"Clases con muestras sin mapeo binario en {root_dir}: {details}. "
                "No se omiten para evitar evaluaciones con cobertura incompleta."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # Cargar el array (128, 128)
        mel_spec = np.load(path)
        # Convertir a tensor y añadir dimensión de canal: (1, 128, 128)
        mel_tensor = torch.from_numpy(mel_spec).unsqueeze(0).float()
        return mel_tensor, torch.tensor(label, dtype=torch.long), path


def make_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=False, generator=None):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "pin_memory": PIN_MEMORY,
        "num_workers": DATALOADER_NUM_WORKERS,
    }
    if generator is not None:
        kwargs["generator"] = generator
    if DATALOADER_NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = DATALOADER_PREFETCH_FACTOR
    return DataLoader(dataset, **kwargs)


# ==========================================
# FUNCIONES DE ENTRENAMIENTO Y EVALUACIÓN
# ==========================================
def run_epoch(loader, model, criterion, optimizer=None, mode=None):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    running_loss, correct, total = 0.0, 0, 0
    tp, fn, fp = 0, 0, 0

    mode = mode or ("TRAIN" if is_train else "EVAL")
    loop = progress(loader, desc=f"{mode}", unit="batch", leave=False)

    with torch.set_grad_enabled(is_train):
        for batch in loop:
            inputs, labels = batch[0], batch[1]
            inputs = inputs.to(device, non_blocking=NON_BLOCKING_TRANSFER)
            labels = labels.to(device, non_blocking=NON_BLOCKING_TRANSFER)

            # 1. Propagación hacia adelante
            outputs = model(inputs)

            # AQUÍ ES DONDE DABA EL ERROR: labels contiene un -1
            loss = criterion(outputs, labels)

            # 2. Propagación hacia atrás (solo en entrenamiento)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            # 3. Estadísticas
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

            loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

    avg_acc = 100. * correct / total if total else 0.0
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)

    return avg_acc, precision, recall, f1_score


def evaluate_epoch_metrics(loader, model, criterion, mode="VALIDATION"):
    model.eval()
    running_loss, total = 0.0, 0
    all_labels, all_probs, all_preds = [], [], []

    loop = progress(loader, desc=mode, unit="batch", leave=False)
    with torch.no_grad():
        for batch in loop:
            inputs, labels = batch[0], batch[1]
            inputs = inputs.to(device, non_blocking=NON_BLOCKING_TRANSFER)
            labels = labels.to(device, non_blocking=NON_BLOCKING_TRANSFER)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            loop.set_postfix(loss=loss.item())

    all_labels = np.asarray(all_labels, dtype=int)
    all_preds = np.asarray(all_preds, dtype=int)
    all_probs = np.asarray(all_probs, dtype=np.float64)
    avg_loss = running_loss / total if total else 0.0

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, all_preds) * 100.0 if total else 0.0,
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
    }
    if len(np.unique(all_labels)) == 2:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        metrics["roc_auc"] = auc(fpr, tpr)
        metrics["pr_auc"] = average_precision_score(all_labels, all_probs)

    return metrics


def metricas_informes(modelo, loader, split_name, excel_file, predictions_file, figure_file, seed=None):
    print("\n" + "=" * 30)
    print(f"GENERATING FINAL REPORT: {split_name.upper()}")
    print("=" * 30)

    # 1. Cargar el mejor modelo guardado
    model = BioCAS_Advanced_CNN().to(device)
    model.load_state_dict(torch.load(modelo, map_location=device))
    model.eval()

    all_preds, all_labels, all_probs, all_paths = [], [], [], []

    with torch.no_grad():
        for batch in progress(loader, desc=f"REPORTE {split_name.upper()}", unit="batch", leave=False):
            images, labels, paths = batch[0], batch[1], batch[2]
            images = images.to(device, non_blocking=NON_BLOCKING_TRANSFER)
            labels = labels.to(device, non_blocking=NON_BLOCKING_TRANSFER)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            # Probabilidades (Softmax) para la clase positiva (Adventicio)
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(paths)

    # 2. Imprimir métricas de texto
    target_names = ["Healthy (Normal)", "Adventitious"]
    print(classification_report(all_labels, all_preds, labels=[0, 1], target_names=target_names, zero_division=0))

    report_dict = classification_report(
        all_labels, all_preds, labels=[0, 1], target_names=target_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    if len(set(all_labels)) == 2:
        fpr, tpr, roc_thresholds = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(all_labels, all_probs)
        pr_auc = average_precision_score(all_labels, all_probs)
    else:
        fpr, tpr, roc_thresholds = np.array([]), np.array([]), np.array([])
        precision_curve, recall_curve, pr_thresholds = np.array([]), np.array([]), np.array([])
        roc_auc, pr_auc = np.nan, np.nan

    df_report = pd.DataFrame(report_dict).transpose()
    df_cm = pd.DataFrame(
        cm,
        index=["True_Healthy", "True_Adventitious"],
        columns=["Pred_Healthy", "Pred_Adventitious"],
    )
    df_predictions = pd.DataFrame({
        "run_id": RUN_ID,
        "experiment_block": EXPERIMENT_BLOCK,
        "dataset_protocol": DATASET_PROTOCOL,
        "model_family": MODEL_FAMILY,
        "training_mode": TRAINING_MODE,
        "classifier_type": CLASSIFIER_TYPE,
        "seed": seed,
        "split": split_name,
        "sample_path": all_paths,
        "y_true": all_labels,
        "prob_adventitious": all_probs,
        "y_pred_threshold_0_5": all_preds,
        "threshold": 0.5,
    })
    os.makedirs(os.path.dirname(predictions_file), exist_ok=True)
    df_predictions.to_csv(predictions_file, index=False)
    print(f">>> Predicciones y probabilidades guardadas en: {predictions_file}")

    nuevo_resultado = {
        "Fecha": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        **EXPERIMENT_METADATA,
        "Seed": seed,
        "Split_Seed": SPLIT_SEED,
        "Training_Seeds": SEED_LIST_LABEL,
        "Split": split_name,
        "Modelo": MODEL_DISPLAY_NAME,
        "Mels": N_MELS,
        "Size": SIZE,
        "Batch_Size": BATCH_SIZE,
        "LR": LEARNING_RATE,
        "Checkpoint_Selection": CHECKPOINT_SELECTION_METRIC,
        "Weight1": WEIGHT1,
        "Weight2": WEIGHT2,
        "Dropout": DROPOUT,
        "Epochs": EPOCHS,
        "Groups": GRUPOS,
        "AUC_ROC": round(roc_auc, 4) if not np.isnan(roc_auc) else np.nan,
        "AUC_PR": round(pr_auc, 4) if not np.isnan(pr_auc) else np.nan,
        "F1-Score_Adv": round(report_dict["Adventitious"]["f1-score"], 4),
        "Recall_Adv": round(report_dict["Adventitious"]["recall"], 4),
        "Precision_Adv": round(report_dict["Adventitious"]["precision"], 4),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    }
    print(nuevo_resultado)

    df_nuevo = pd.DataFrame([nuevo_resultado])

    sheet_name = split_name[:31]
    GAP = 3
    try:
        os.makedirs(os.path.dirname(excel_file), exist_ok=True)
        # Si el archivo no existe, lo creamos normalmente
        if not os.path.exists(excel_file):
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                df_report.to_excel(writer, sheet_name=sheet_name)
                start_row_cm = len(df_report) + GAP
                df_cm.to_excel(writer, sheet_name=sheet_name, startrow=start_row_cm)
                start_row_nuevo = start_row_cm + len(df_cm) + GAP
                df_nuevo.to_excel(writer, sheet_name=sheet_name, startrow=start_row_nuevo)
        else:
            # Si existe, añadimos o reemplazamos la hoja
            with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                df_report.to_excel(writer, sheet_name=sheet_name)
                start_row_cm = len(df_report) + GAP
                df_cm.to_excel(writer, sheet_name=sheet_name, startrow=start_row_cm)
                start_row_nuevo = start_row_cm + len(df_cm) + GAP
                df_nuevo.to_excel(writer, sheet_name=sheet_name, startrow=start_row_nuevo)

        print(f">>> Informe exportado exitosamente a: {excel_file}")
    except Exception as e:
        print(f"Error crítico al guardar Excel: {e}")

    # Graficar matriz, ROC y precision-recall.
    plt.figure(figsize=(15, 5))
    # Subplot 1: Matriz
    plt.subplot(1, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    # Subplot 2: Curva ROC
    plt.subplot(1, 3, 2)
    if len(fpr):
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.legend(loc="lower right")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("ROC Curve")
    # Subplot 3: Precision-Recall
    plt.subplot(1, 3, 3)
    if len(precision_curve):
        plt.plot(recall_curve, precision_curve, color='darkgreen', lw=2, label=f'PR AUC = {pr_auc:.2f}')
        plt.legend(loc="lower left")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    os.makedirs(os.path.dirname(figure_file), exist_ok=True)
    plt.savefig(figure_file, dpi=300)
    plt.close()
    print(f">>> Figura de análisis guardada en: {figure_file}")

    return nuevo_resultado


def clip_probabilities(probabilities, eps=1e-6):
    return np.clip(np.asarray(probabilities, dtype=np.float64), eps, 1.0 - eps)


def probability_to_logit(probabilities):
    probabilities = clip_probabilities(probabilities)
    return np.log(probabilities / (1.0 - probabilities))


def sigmoid_np(values):
    values = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-values))


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
    candidate_temperatures = np.concatenate([
        np.linspace(0.20, 2.00, 91),
        np.linspace(2.05, 8.00, 120),
    ])
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


def analizar_thresholds_y_calibracion(validation_predictions_file, test_predictions_file, output_dir, run_base_name, seed):
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

    for calibration_name, payload in progress(calibrated.items(), desc=f"Calibración seed {seed}", unit="modo", leave=False):
        val_scores = clip_probabilities(payload["validation"])
        test_scores = clip_probabilities(payload["test"])
        validation_enriched[f"prob_{calibration_name}"] = val_scores
        test_enriched[f"prob_{calibration_name}"] = test_scores

        thresholds = select_thresholds_from_validation(y_val, val_scores)
        for threshold_name, threshold in thresholds.items():
            extra = payload.get("extra", {})
            rows.append(binary_metrics_for_threshold(
                y_val, val_scores, threshold, "validation", calibration_name, threshold_name, seed=seed, extra=extra
            ))
            rows.append(binary_metrics_for_threshold(
                y_test, test_scores, threshold, "test", calibration_name, threshold_name, seed=seed, extra=extra
            ))

    os.makedirs(output_dir, exist_ok=True)
    file_prefix = f"{run_base_name}_seed{seed}_" if run_base_name else ""
    metrics_file = os.path.join(output_dir, f"{file_prefix}threshold_calibration_metrics.csv")
    validation_calibrated_file = os.path.join(output_dir, f"{file_prefix}predicciones_validation_calibradas.csv")
    test_calibrated_file = os.path.join(output_dir, f"{file_prefix}predicciones_test_calibradas.csv")

    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(metrics_file, index=False)
    validation_enriched.to_csv(validation_calibrated_file, index=False)
    test_enriched.to_csv(test_calibrated_file, index=False)

    print(f">>> Métricas de thresholds/calibración guardadas en: {metrics_file}")
    print(f">>> Predicciones calibradas validation guardadas en: {validation_calibrated_file}")
    print(f">>> Predicciones calibradas test guardadas en: {test_calibrated_file}")

    cols = ["Calibration", "Threshold_Strategy", "Split", "Threshold", "F1_Adv", "Recall_Adv", "Specificity", "Precision_Adv", "AUC_ROC", "AUC_PR"]
    print("\nResumen thresholds/calibración:")
    print(df_metrics[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return df_metrics


def run_preprocessing():
    validate_implemented_experiment()

    splits = preparar_splits_sprsound()
    segmentar_carpetas(splits)
    auditar_dataset_biocas(OUTPUT_BASE_DIR, CARPETAS_A_REVISAR)

    if PADDING_MODE == 1:
        target_w_final = calcular_ancho_percentil(OUTPUT_BASE_DIR, ["segmentado_train"], DURACION_PERCENTIL)
    else:
        target_w_final = SIZE
    procesar_y_guardar(OUTPUT_BASE_DIR, PATH_SALIDA_NPY, CARPETAS_A_REVISAR, target_w_final)
    return target_w_final


def split_numpy_ready(split_path):
    if not os.path.isdir(split_path):
        return False
    for _, _, files in os.walk(split_path):
        if any(name.endswith(".npy") for name in files):
            return True
    return False


def assert_preprocessed_data_ready():
    required = {
        "train": TRAIN_PATH,
        "validation": VALIDATION_PATH,
        "test": TEST_PATH,
    }
    missing = [name for name, path in required.items() if not split_numpy_ready(path)]
    if missing:
        raise FileNotFoundError(
            "Faltan datos preprocesados para: "
            + ", ".join(missing)
            + ". Ejecuta primero el script de preprocesado."
        )


def infer_target_width_from_preprocessed_data():
    if PADDING_MODE == 0:
        return SIZE
    for split_path in (TRAIN_PATH, VALIDATION_PATH, TEST_PATH):
        for root, _, files in os.walk(split_path):
            for name in files:
                if name.endswith(".npy"):
                    return int(np.load(os.path.join(root, name), mmap_mode="r").shape[-1])
    return None


def run_training():
    assert_preprocessed_data_ready()
    os.makedirs(SUMMARY_OUTPUT_DIR, exist_ok=True)
    print(f">>> Carpeta raíz del experimento: {MODEL_OUTPUT_DIR}")
    print(f">>> Resúmenes multi-seed: {SUMMARY_OUTPUT_DIR}")
    print(f">>> Seeds de entrenamiento: {SEEDS}")
    print(f">>> Seed de split patient-independent: {SPLIT_SEED}")

    resultados_finales = []
    resultados_threshold_calibracion = []
    for seed in progress(SEEDS, desc="Seeds de entrenamiento", unit="seed"):
        set_seed(seed)
        seed_dirs = build_seed_output_dirs(seed)
        train_ds = BioCASNumpyDataset(root_dir=TRAIN_PATH)
        validation_ds = BioCASNumpyDataset(root_dir=VALIDATION_PATH)
        test_ds = BioCASNumpyDataset(root_dir=TEST_PATH)

        train_generator = torch.Generator()
        train_generator.manual_seed(seed)
        train_loader = make_dataloader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=train_generator)
        validation_loader = make_dataloader(validation_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = make_dataloader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        print(
            f"Dataset NPY cargado para seed {seed}: "
            f"{len(train_ds)} train, {len(validation_ds)} validation, {len(test_ds)} test."
        )
        write_seed_manifest(
            seed=seed,
            seed_dirs=seed_dirs,
            dataset_sizes={
                "train": len(train_ds),
                "validation": len(validation_ds),
                "test": len(test_ds),
            },
        )

        model_path = os.path.join(seed_dirs["checkpoints"], f"best_validation_{CHECKPOINT_SELECTION_METRIC}.pth")
        excel_path = os.path.join(seed_dirs["reports"], "metricas.xlsx")
        validation_predictions_path = os.path.join(seed_dirs["predictions"], "predicciones_validation.csv")
        test_predictions_path = os.path.join(seed_dirs["predictions"], "predicciones_test.csv")
        validation_figure_path = os.path.join(seed_dirs["figures"], "analisis_validation.png")
        test_figure_path = os.path.join(seed_dirs["figures"], "analisis_test.png")

        model = BioCAS_Advanced_CNN().to(device)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([WEIGHT1, WEIGHT2]).to(device))
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        if CHECKPOINT_SELECTION_METRIC not in {"f1", "roc_auc", "pr_auc"}:
            raise ValueError(f"CHECKPOINT_SELECTION_METRIC no soportada: {CHECKPOINT_SELECTION_METRIC}")

        best_val_score = -np.inf
        best_epoch = None
        for epoch in progress(range(1, EPOCHS + 1), desc=f"Epochs seed {seed}", unit="epoch"):

            print(f"\n--- SPRSound | Seed {seed} | Epoch {epoch}/{EPOCHS} ---")
            run_epoch(train_loader, model, criterion, optimizer, mode="TRAIN")

            val_metrics = evaluate_epoch_metrics(validation_loader, model, criterion, mode="VALIDATION")
            val_score = val_metrics[CHECKPOINT_SELECTION_METRIC]
            print(f"\n[RESULTADOS VALIDATION SPRSound]")
            print(f"Loss:      {val_metrics['loss']:.4f}")
            print(f"Exactitud: {val_metrics['accuracy']:.2f}%")
            print(f"Precisión: {val_metrics['precision']:.4f}")
            print(f"Recall:    {val_metrics['recall']:.4f} (Sensibilidad)")
            print(f"F1-Score:  {val_metrics['f1']:.4f}")
            print(f"ROC AUC:   {val_metrics['roc_auc']:.4f}")
            print(f"PR AUC:    {val_metrics['pr_auc']:.4f}")

            if not np.isnan(val_score) and val_score > best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                torch.save(model.state_dict(), model_path)
                print(
                    f"Nuevo mejor checkpoint por validation. "
                    f"Epoch: {epoch} | {CHECKPOINT_SELECTION_METRIC}: {val_score:.4f}"
                )

        print(
            f"\nMejor checkpoint seed {seed}: epoch {best_epoch}, "
            f"{CHECKPOINT_SELECTION_METRIC} validation {best_val_score:.4f}"
        )
        resultados_finales.append(metricas_informes(
            modelo=model_path,
            loader=validation_loader,
            split_name="validation",
            excel_file=excel_path,
            predictions_file=validation_predictions_path,
            figure_file=validation_figure_path,
            seed=seed,
        ))
        resultados_finales.append(metricas_informes(
            modelo=model_path,
            loader=test_loader,
            split_name="test",
            excel_file=excel_path,
            predictions_file=test_predictions_path,
            figure_file=test_figure_path,
            seed=seed,
        ))
        resultados_threshold_calibracion.append(analizar_thresholds_y_calibracion(
            validation_predictions_file=validation_predictions_path,
            test_predictions_file=test_predictions_path,
            output_dir=seed_dirs["calibration"],
            run_base_name="",
            seed=seed,
        ))
        if device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    df_resultados = pd.DataFrame(resultados_finales)
    resumen_path = os.path.join(SUMMARY_OUTPUT_DIR, "resumen_seeds.csv")
    df_resultados.to_csv(resumen_path, index=False)
    print(f">>> Resumen por seed guardado en: {resumen_path}")

    if resultados_threshold_calibracion:
        df_thresholds = pd.concat(resultados_threshold_calibracion, ignore_index=True)
        thresholds_resumen_path = os.path.join(SUMMARY_OUTPUT_DIR, "threshold_calibration_resumen_seeds.csv")
        df_thresholds.to_csv(thresholds_resumen_path, index=False)
        print(f">>> Resumen thresholds/calibración por seed guardado en: {thresholds_resumen_path}")

    if len(SEEDS) > 1:
        metric_cols = ["AUC_ROC", "AUC_PR", "F1-Score_Adv", "Recall_Adv", "Precision_Adv", "TN", "FP", "FN", "TP"]
        resumen_media_std_path = os.path.join(SUMMARY_OUTPUT_DIR, "media_std_por_split.csv")
        df_resultados.groupby("Split")[metric_cols].agg(["mean", "std"]).to_csv(resumen_media_std_path)
        print(f">>> Media y desviación estándar por split guardadas en: {resumen_media_std_path}")

        if resultados_threshold_calibracion:
            threshold_metric_cols = ["Accuracy", "AUC_ROC", "AUC_PR", "Brier", "LogLoss", "F1_Adv", "Recall_Adv", "Specificity", "Precision_Adv"]
            thresholds_media_std_path = os.path.join(SUMMARY_OUTPUT_DIR, "threshold_calibration_media_std.csv")
            df_thresholds.groupby(["Calibration", "Threshold_Strategy", "Split"])[threshold_metric_cols].agg(["mean", "std"]).to_csv(thresholds_media_std_path)
            print(f">>> Media/std thresholds-calibración guardadas en: {thresholds_media_std_path}")


def main():
    validate_implemented_experiment()
    assert_preprocessed_data_ready()
    target_w_final = infer_target_width_from_preprocessed_data()
    write_run_manifest(target_width=target_w_final)
    run_training()


if __name__ == "__main__":
    main()
