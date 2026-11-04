from datetime import datetime
import glob
import json
import os
import zlib

# Localiza la raíz del proyecto para mantener rutas relativas a Fix_Paper1.
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
_TFHUB_CACHE_DIR = os.path.join(_CACHE_DIR, "tfhub")
_DEFAULT_CPU_THREAD_COUNT = str(min(8, os.cpu_count() or 4))
os.environ.setdefault("XDG_CACHE_HOME", _CACHE_DIR)
os.environ.setdefault("TFHUB_CACHE_DIR", _TFHUB_CACHE_DIR)
os.environ.setdefault("OMP_NUM_THREADS", _DEFAULT_CPU_THREAD_COUNT)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", os.environ["OMP_NUM_THREADS"])
os.environ.setdefault("NUMEXPR_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", os.environ["OMP_NUM_THREADS"])
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
os.makedirs(os.environ["TFHUB_CACHE_DIR"], exist_ok=True)

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

try:
    import tensorflow_hub as hub
except ModuleNotFoundError as exc:
    if exc.name == "pkg_resources":
        raise RuntimeError(
            "tensorflow_hub no es compatible con setuptools>=82 en este entorno. "
            "Fija setuptools<82 en fix-paper1 y actualiza el entorno antes de ejecutar YAMNet. "
            "Comando recomendado: conda env update -n fix-paper1 -f environment.yml"
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


DATASET_NAME = os.environ.get("DATASET_NAME", "0 SPRSound")
DATASET_LABEL = os.environ.get("DATASET_LABEL", slugify(DATASET_NAME))
MODEL_HANDLE = os.environ.get("YAMNET_MODEL_HANDLE", "https://tfhub.dev/google/yamnet/1")
SPLIT_SEED = int(os.environ.get("SPLIT_SEED", 42))
EMBEDDING_SAMPLE_RATE = int(os.environ.get("YAMNET_SAMPLE_RATE", 16000))
MIN_YAMNET_SECONDS = float(os.environ.get("YAMNET_MIN_SECONDS", 0.975))
MIN_YAMNET_SAMPLES = int(round(EMBEDDING_SAMPLE_RATE * MIN_YAMNET_SECONDS))
POOLING_MODE = os.environ.get("POOLING_MODE", "mean")
# R1: relleno de eventos cortos. zero = comportamiento original; loop = repetir el evento;
# noise = ruido gaussiano a NOISE_RELATIVE_DB respecto al RMS del propio evento.
PAD_MODE = os.environ.get("PAD_MODE", "zero").strip().lower()
NOISE_RELATIVE_DB = float(os.environ.get("NOISE_RELATIVE_DB", -40.0))
NOISE_SEED = int(os.environ.get("NOISE_SEED", 12345))
if PAD_MODE not in {"zero", "loop", "noise"}:
    raise ValueError(f"PAD_MODE no soportado: {PAD_MODE}")
EMBEDDING_DIM = 1024
SEGMENTED_WAV_DIR = os.environ.get(
    "SEGMENTED_WAV_DIR",
    os.path.join(_PROJECT_DIR, "ficheros", "1 Segmentados_WAV", DATASET_NAME, f"seed_{SPLIT_SEED}"),
)
EMBEDDINGS_OUTPUT_DIR = os.environ.get(
    "EMBEDDINGS_OUTPUT_DIR",
    os.path.join(
        _PROJECT_DIR,
        "ficheros",
        "3 Embeddings YAMNet",
        f"{DATASET_NAME}_seed{SPLIT_SEED}_yamnet_tfhub_mean_pool_1024",
    ),
)
SPLIT_DIRS = {
    "train": "segmentado_train",
    "validation": "segmentado_validation",
    "test": "segmentado_test",
}
EXPERIMENT_NOTES = os.environ.get(
    "EXPERIMENT_NOTES",
    "Embeddings YAMNet frozen extraidos desde los segmentados WAV ya generados por el protocolo actual.",
)

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


def assert_segmented_wavs_ready():
    missing = []
    for split_name, split_dir_name in SPLIT_DIRS.items():
        split_dir = os.path.join(SEGMENTED_WAV_DIR, split_dir_name)
        if not os.path.isdir(split_dir):
            missing.append(f"{split_name}:{split_dir}")
            continue
        has_wavs = bool(glob.glob(os.path.join(split_dir, "*", "*.wav")))
        if not has_wavs:
            missing.append(f"{split_name}:{split_dir}")
    if missing:
        raise FileNotFoundError(
            "Faltan segmentados WAV para YAMNet en: "
            + ", ".join(missing)
            + ". Ejecuta antes el preprocesado CNN correspondiente."
        )


def load_segment_paths(split_dir):
    return sorted(glob.glob(os.path.join(split_dir, "*", "*.wav")))


def infer_binary_label(segment_path):
    label_name = os.path.basename(os.path.dirname(segment_path))
    if label_name not in EVENT_TO_BINARY:
        raise ValueError(f"Etiqueta de segmento no soportada para YAMNet: {label_name}")
    return label_name, EVENT_TO_BINARY[label_name]


def infer_source_dataset(segment_path):
    basename = os.path.basename(segment_path)
    if basename.startswith("sprsound__"):
        return "sprsound"
    if basename.startswith("icbhi__"):
        return "icbhi"
    return DATASET_LABEL


def pad_waveform_for_yamnet(waveform, segment_path=None):
    if len(waveform) >= MIN_YAMNET_SAMPLES:
        return waveform
    pad_width = (0, MIN_YAMNET_SAMPLES - len(waveform))
    if PAD_MODE == "loop":
        return np.pad(waveform, pad_width, mode="wrap")
    if PAD_MODE == "noise":
        # Semilla por fichero para que el relleno sea reproducible e independiente del orden.
        file_seed = (NOISE_SEED + zlib.crc32(os.path.basename(segment_path or "").encode("utf-8"))) % (2**32)
        rng = np.random.default_rng(file_seed)
        event_rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
        noise_rms = event_rms * 10.0 ** (NOISE_RELATIVE_DB / 20.0)
        noise = rng.normal(0.0, noise_rms, size=pad_width[1]).astype(waveform.dtype)
        return np.concatenate([waveform, noise])
    return np.pad(waveform, pad_width, mode="constant")


def load_yamnet_model():
    tf.get_logger().setLevel("ERROR")
    tf.config.threading.set_intra_op_parallelism_threads(int(os.environ["TF_NUM_INTRAOP_THREADS"]))
    tf.config.threading.set_inter_op_parallelism_threads(int(os.environ["TF_NUM_INTEROP_THREADS"]))
    return hub.load(MODEL_HANDLE)


def pool_embeddings(frame_embeddings):
    if frame_embeddings.ndim != 2 or frame_embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"Embedding YAMNet inesperado: {frame_embeddings.shape}")
    if POOLING_MODE != "mean":
        raise ValueError(f"POOLING_MODE no soportado: {POOLING_MODE}")
    return np.mean(frame_embeddings, axis=0, dtype=np.float32)


def extract_segment_embedding(model, segment_path):
    waveform, _ = librosa.load(segment_path, sr=EMBEDDING_SAMPLE_RATE, mono=True)
    if waveform.size == 0:
        raise ValueError("Waveform vacio")
    original_samples = int(waveform.shape[0])
    waveform = pad_waveform_for_yamnet(waveform, segment_path).astype(np.float32, copy=False)
    scores, embeddings, _ = model(tf.convert_to_tensor(waveform, dtype=tf.float32))
    embeddings_np = embeddings.numpy()
    pooled = pool_embeddings(embeddings_np)
    metadata = {
        "original_duration_seconds": float(original_samples / EMBEDDING_SAMPLE_RATE),
        "padded": bool(original_samples < MIN_YAMNET_SAMPLES),
        "pad_mode": PAD_MODE,
        "duration_seconds_resampled": float(waveform.shape[0] / EMBEDDING_SAMPLE_RATE),
        "yamnet_frames": int(embeddings_np.shape[0]),
        "mean_score_max": float(np.max(np.mean(scores.numpy(), axis=0))),
    }
    return pooled, metadata


def save_split_artifacts(split_name, embeddings, labels, metadata_rows):
    os.makedirs(EMBEDDINGS_OUTPUT_DIR, exist_ok=True)
    prefix = SPLIT_DIRS[split_name]
    npz_path = os.path.join(EMBEDDINGS_OUTPUT_DIR, f"{prefix}_embeddings.npz")
    csv_path = os.path.join(EMBEDDINGS_OUTPUT_DIR, f"{prefix}_metadata.csv")
    np.savez_compressed(
        npz_path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
    )
    pd.DataFrame(metadata_rows).to_csv(csv_path, index=False)
    return npz_path, csv_path


def process_split(model, split_name):
    split_dir = os.path.join(SEGMENTED_WAV_DIR, SPLIT_DIRS[split_name])
    segment_paths = load_segment_paths(split_dir)
    embeddings = []
    labels = []
    metadata_rows = []
    errors = []

    print(f"\n--- Extrayendo embeddings YAMNet: {split_name} ({len(segment_paths)} segmentos) ---")
    for segment_path in progress(segment_paths, desc=f"YAMNet {split_name}", unit="seg"):
        try:
            label_name, label_value = infer_binary_label(segment_path)
            embedding, extra = extract_segment_embedding(model, segment_path)
            embeddings.append(embedding)
            labels.append(label_value)
            metadata_rows.append({
                "split": split_name,
                "sample_path": segment_path,
                "segment_file": os.path.basename(segment_path),
                "label_name": label_name,
                "label_binary": label_value,
                "source_dataset": infer_source_dataset(segment_path),
                **extra,
            })
        except Exception as exc:
            errors.append({
                "split": split_name,
                "sample_path": segment_path,
                "error": str(exc),
            })

    npz_path, csv_path = save_split_artifacts(split_name, embeddings, labels, metadata_rows)
    summary = {
        "split": split_name,
        "n_segments": int(len(labels)),
        "n_negative": int(np.sum(np.asarray(labels) == 0)) if labels else 0,
        "n_positive": int(np.sum(np.asarray(labels) == 1)) if labels else 0,
        "npz_path": npz_path,
        "metadata_csv": csv_path,
        "n_errors": len(errors),
    }
    return summary, errors


def write_manifest(split_summaries, all_errors):
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": DATASET_NAME,
        "dataset_label": DATASET_LABEL,
        "split_seed": SPLIT_SEED,
        "segmented_wav_dir": SEGMENTED_WAV_DIR,
        "embeddings_output_dir": EMBEDDINGS_OUTPUT_DIR,
        "model_handle": MODEL_HANDLE,
        "embedding_sample_rate": EMBEDDING_SAMPLE_RATE,
        "minimum_yamnet_seconds": MIN_YAMNET_SECONDS,
        "minimum_yamnet_samples": MIN_YAMNET_SAMPLES,
        "pooling_mode": POOLING_MODE,
        "pad_mode": PAD_MODE,
        "noise_relative_db": NOISE_RELATIVE_DB if PAD_MODE == "noise" else None,
        "noise_seed": NOISE_SEED if PAD_MODE == "noise" else None,
        "embedding_dim": EMBEDDING_DIM,
        "notes": EXPERIMENT_NOTES,
        "splits": split_summaries,
        "errors": all_errors,
    }
    manifest_path = os.path.join(EMBEDDINGS_OUTPUT_DIR, "embedding_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f">>> Manifiesto de embeddings YAMNet guardado en: {manifest_path}")


def main():
    assert_segmented_wavs_ready()
    print(f">>> Directorio segmentado de entrada: {SEGMENTED_WAV_DIR}")
    print(f">>> Directorio de salida de embeddings: {EMBEDDINGS_OUTPUT_DIR}")
    print(f">>> Modelo YAMNet: {MODEL_HANDLE}")
    model = load_yamnet_model()

    split_summaries = {}
    all_errors = []
    for split_name in ("train", "validation", "test"):
        summary, errors = process_split(model, split_name)
        split_summaries[split_name] = summary
        all_errors.extend(errors)
        print(
            f">>> {split_name}: {summary['n_segments']} segmentos | "
            f"normal={summary['n_negative']} | adventitious={summary['n_positive']} | "
            f"errors={summary['n_errors']}"
        )

    write_manifest(split_summaries, all_errors)
    if all_errors:
        print(f">>> Aviso: hubo {len(all_errors)} segmentos con error. Revisa embedding_manifest.json.")


if __name__ == "__main__":
    main()
