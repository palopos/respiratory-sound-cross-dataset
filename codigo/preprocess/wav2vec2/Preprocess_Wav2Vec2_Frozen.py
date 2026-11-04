from datetime import datetime
import glob
import json
import os

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
_DEFAULT_CPU_THREAD_COUNT = str(min(8, os.cpu_count() or 4))
os.environ.setdefault("XDG_CACHE_HOME", _CACHE_DIR)
os.environ.setdefault("HF_HOME", _HF_CACHE_DIR)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("OMP_NUM_THREADS", _DEFAULT_CPU_THREAD_COUNT)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", os.environ["OMP_NUM_THREADS"])
os.environ.setdefault("NUMEXPR_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    from transformers import AutoFeatureExtractor, AutoModel
except ModuleNotFoundError as exc:
    if exc.name in {"transformers", "huggingface_hub", "tokenizers", "safetensors"}:
        raise RuntimeError(
            "Faltan dependencias de wav2vec2 en fix-paper1. "
            "Actualiza el entorno antes de ejecutar este script. "
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


def resolve_device_name():
    requested = os.environ.get("WAV2VEC2_DEVICE", "auto").strip().lower()
    mps_available = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    if requested == "auto":
        return "mps" if mps_available else "cpu"
    if requested == "mps":
        if not mps_available:
            raise RuntimeError("WAV2VEC2_DEVICE=mps pero MPS no esta disponible en este entorno.")
        return "mps"
    if requested == "cpu":
        return "cpu"
    raise ValueError(f"WAV2VEC2_DEVICE no soportado: {requested}")


DATASET_NAME = os.environ.get("DATASET_NAME", "0 SPRSound")
DATASET_LABEL = os.environ.get("DATASET_LABEL", slugify(DATASET_NAME))
MODEL_ID = os.environ.get("WAV2VEC2_MODEL_ID", "facebook/wav2vec2-base")
SPLIT_SEED = int(os.environ.get("SPLIT_SEED", 42))
EMBEDDING_SAMPLE_RATE = int(os.environ.get("WAV2VEC2_SAMPLE_RATE", 16000))
POOLING_MODE = os.environ.get("POOLING_MODE", "mean")
DEVICE_NAME = resolve_device_name()
BATCH_SIZE = int(os.environ.get("WAV2VEC2_BATCH_SIZE", 8))
LOCAL_FILES_ONLY = os.environ.get("HF_LOCAL_FILES_ONLY", "0") == "1"
USE_SAFETENSORS = os.environ.get("WAV2VEC2_USE_SAFETENSORS", "1") == "1"
SEGMENTED_WAV_DIR = os.environ.get(
    "SEGMENTED_WAV_DIR",
    os.path.join(_PROJECT_DIR, "ficheros", "1 Segmentados_WAV", DATASET_NAME, f"seed_{SPLIT_SEED}"),
)
EMBEDDINGS_OUTPUT_DIR = os.environ.get(
    "EMBEDDINGS_OUTPUT_DIR",
    os.path.join(
        _PROJECT_DIR,
        "ficheros",
        "3 Embeddings wav2vec2",
        f"{DATASET_NAME}_seed{SPLIT_SEED}_wav2vec2_hf_mean_pool_768",
    ),
)
SPLIT_DIRS = {
    "train": "segmentado_train",
    "validation": "segmentado_validation",
    "test": "segmentado_test",
}
EXPERIMENT_NOTES = os.environ.get(
    "EXPERIMENT_NOTES",
    "Embeddings wav2vec2 frozen extraidos desde los segmentados WAV ya generados por el protocolo actual.",
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
            "Faltan segmentados WAV para wav2vec2 en: "
            + ", ".join(missing)
            + ". Ejecuta antes el preprocesado CNN correspondiente."
        )


def load_segment_paths(split_dir):
    return sorted(glob.glob(os.path.join(split_dir, "*", "*.wav")))


def infer_binary_label(segment_path):
    label_name = os.path.basename(os.path.dirname(segment_path))
    if label_name not in EVENT_TO_BINARY:
        raise ValueError(f"Etiqueta de segmento no soportada para wav2vec2: {label_name}")
    return label_name, EVENT_TO_BINARY[label_name]


def infer_source_dataset(segment_path):
    basename = os.path.basename(segment_path)
    if basename.startswith("sprsound__"):
        return "sprsound"
    if basename.startswith("icbhi__"):
        return "icbhi"
    return DATASET_LABEL


def load_waveform(segment_path):
    waveform, _ = librosa.load(segment_path, sr=EMBEDDING_SAMPLE_RATE, mono=True)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0:
        raise ValueError("Waveform vacio")
    return waveform


def configure_torch_runtime():
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass


def load_model_and_extractor():
    configure_torch_runtime()
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, local_files_only=LOCAL_FILES_ONLY)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        local_files_only=LOCAL_FILES_ONLY,
        use_safetensors=USE_SAFETENSORS,
    )
    model.eval()
    model.to(torch.device(DEVICE_NAME))
    return feature_extractor, model


def compute_min_input_samples(model):
    candidate_lengths = torch.arange(1, EMBEDDING_SAMPLE_RATE + 1, dtype=torch.long)
    output_lengths = model._get_feat_extract_output_lengths(candidate_lengths)
    valid_positions = (output_lengths >= 1).nonzero(as_tuple=False)
    if valid_positions.numel() == 0:
        raise RuntimeError("No se pudo calcular la longitud minima de entrada para wav2vec2.")
    return int(valid_positions[0].item() + 1)


def pad_waveform(waveform, min_input_samples):
    if waveform.shape[0] >= min_input_samples:
        return waveform, int(waveform.shape[0]), False
    padded = np.pad(waveform, (0, min_input_samples - waveform.shape[0]), mode="constant")
    return padded.astype(np.float32, copy=False), int(min_input_samples), True


def use_attention_mask_for_model(model):
    return str(getattr(model.config, "feat_extract_norm", "")).strip().lower() == "layer"


def prepare_segment_record(segment_path, min_input_samples):
    label_name, label_value = infer_binary_label(segment_path)
    waveform = load_waveform(segment_path)
    original_samples = int(waveform.shape[0])
    waveform, effective_samples, was_padded = pad_waveform(waveform, min_input_samples)
    return {
        "segment_path": segment_path,
        "segment_file": os.path.basename(segment_path),
        "label_name": label_name,
        "label_binary": label_value,
        "source_dataset": infer_source_dataset(segment_path),
        "waveform": waveform,
        "original_samples": original_samples,
        "effective_samples": effective_samples,
        "was_padded": was_padded,
        "duration_seconds_resampled": float(original_samples / EMBEDDING_SAMPLE_RATE),
    }


def encode_records(feature_extractor, records):
    return feature_extractor(
        [record["waveform"] for record in records],
        sampling_rate=EMBEDDING_SAMPLE_RATE,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )


def pool_hidden_states(hidden_states, output_lengths):
    if POOLING_MODE != "mean":
        raise ValueError(f"POOLING_MODE no soportado: {POOLING_MODE}")
    pooled_embeddings = []
    frame_counts = []
    hidden_states = hidden_states.detach().cpu()
    max_frames = hidden_states.shape[1]
    for idx in range(hidden_states.shape[0]):
        valid_frames = int(max(1, min(max_frames, int(output_lengths[idx]))))
        pooled_embeddings.append(
            hidden_states[idx, :valid_frames].mean(dim=0).numpy().astype(np.float32, copy=False)
        )
        frame_counts.append(valid_frames)
    return pooled_embeddings, frame_counts


def extract_batch_embeddings(feature_extractor, model, records, pass_attention_mask_to_model):
    encoded = encode_records(feature_extractor, records)
    input_values = encoded["input_values"].to(torch.device(DEVICE_NAME))
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        valid_input_lengths = attention_mask.sum(dim=1).to(torch.long)
    else:
        valid_input_lengths = torch.as_tensor(
            [record["effective_samples"] for record in records],
            dtype=torch.long,
        )

    model_kwargs = {"input_values": input_values}
    if pass_attention_mask_to_model and attention_mask is not None:
        model_kwargs["attention_mask"] = attention_mask.to(torch.device(DEVICE_NAME))

    with torch.inference_mode():
        outputs = model(**model_kwargs)
    hidden_states = outputs.last_hidden_state
    output_lengths = model._get_feat_extract_output_lengths(valid_input_lengths.cpu()).tolist()
    pooled_embeddings, frame_counts = pool_hidden_states(hidden_states, output_lengths)

    rows = []
    for record, embedding, frame_count in zip(records, pooled_embeddings, frame_counts):
        rows.append({
            "embedding": embedding,
            "metadata": {
                "split": None,
                "sample_path": record["segment_path"],
                "segment_file": record["segment_file"],
                "label_name": record["label_name"],
                "label_binary": record["label_binary"],
                "source_dataset": record["source_dataset"],
                "duration_seconds_resampled": record["duration_seconds_resampled"],
                "input_samples_original": record["original_samples"],
                "input_samples_effective": record["effective_samples"],
                "input_was_padded": record["was_padded"],
                "wav2vec2_frames": int(frame_count),
            },
        })
    return rows


def error_row(split_name, segment_path, exc):
    return {
        "split": split_name,
        "sample_path": segment_path,
        "error": str(exc),
    }


def process_batch(feature_extractor, model, split_name, batch_paths, min_input_samples, pass_attention_mask_to_model):
    ready_records = []
    errors = []
    for segment_path in batch_paths:
        try:
            ready_records.append(prepare_segment_record(segment_path, min_input_samples))
        except Exception as exc:
            errors.append(error_row(split_name, segment_path, exc))

    if not ready_records:
        return [], [], [], errors

    try:
        batch_rows = extract_batch_embeddings(feature_extractor, model, ready_records, pass_attention_mask_to_model)
    except Exception as batch_exc:
        batch_rows = []
        fallback_errors = []
        for record in ready_records:
            try:
                batch_rows.extend(
                    extract_batch_embeddings(
                        feature_extractor,
                        model,
                        [record],
                        pass_attention_mask_to_model,
                    )
                )
            except Exception as single_exc:
                fallback_errors.append(
                    error_row(split_name, record["segment_path"], f"{batch_exc} | fallback: {single_exc}")
                )
        errors.extend(fallback_errors)

    embeddings = []
    labels = []
    metadata_rows = []
    for row in batch_rows:
        row["metadata"]["split"] = split_name
        embeddings.append(row["embedding"])
        labels.append(row["metadata"]["label_binary"])
        metadata_rows.append(row["metadata"])
    return embeddings, labels, metadata_rows, errors


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


def process_split(feature_extractor, model, split_name, min_input_samples, pass_attention_mask_to_model):
    split_dir = os.path.join(SEGMENTED_WAV_DIR, SPLIT_DIRS[split_name])
    segment_paths = load_segment_paths(split_dir)
    embeddings = []
    labels = []
    metadata_rows = []
    errors = []

    print(f"\n--- Extrayendo embeddings wav2vec2: {split_name} ({len(segment_paths)} segmentos) ---")
    batch_starts = range(0, len(segment_paths), BATCH_SIZE)
    for start_idx in progress(batch_starts, desc=f"wav2vec2 {split_name}", unit="batch"):
        batch_paths = segment_paths[start_idx:start_idx + BATCH_SIZE]
        batch_embeddings, batch_labels, batch_metadata, batch_errors = process_batch(
            feature_extractor=feature_extractor,
            model=model,
            split_name=split_name,
            batch_paths=batch_paths,
            min_input_samples=min_input_samples,
            pass_attention_mask_to_model=pass_attention_mask_to_model,
        )
        embeddings.extend(batch_embeddings)
        labels.extend(batch_labels)
        metadata_rows.extend(batch_metadata)
        errors.extend(batch_errors)

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
    if DEVICE_NAME == "mps":
        torch.mps.empty_cache()
    return summary, errors


def write_manifest(split_summaries, all_errors, model, feature_extractor, min_input_samples, pass_attention_mask_to_model):
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": DATASET_NAME,
        "dataset_label": DATASET_LABEL,
        "split_seed": SPLIT_SEED,
        "segmented_wav_dir": SEGMENTED_WAV_DIR,
        "embeddings_output_dir": EMBEDDINGS_OUTPUT_DIR,
        "model_id": MODEL_ID,
        "embedding_sample_rate": EMBEDDING_SAMPLE_RATE,
        "device": DEVICE_NAME,
        "batch_size": BATCH_SIZE,
        "local_files_only": LOCAL_FILES_ONLY,
        "use_safetensors": USE_SAFETENSORS,
        "minimum_input_samples": min_input_samples,
        "pooling_mode": POOLING_MODE,
        "embedding_dim": int(getattr(model.config, "hidden_size", 0)),
        "feature_extractor_sampling_rate": int(getattr(feature_extractor, "sampling_rate", EMBEDDING_SAMPLE_RATE)),
        "model_uses_attention_mask": pass_attention_mask_to_model,
        "notes": EXPERIMENT_NOTES,
        "splits": split_summaries,
        "errors": all_errors,
    }
    manifest_path = os.path.join(EMBEDDINGS_OUTPUT_DIR, "embedding_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f">>> Manifiesto de embeddings wav2vec2 guardado en: {manifest_path}")


def main():
    assert_segmented_wavs_ready()
    print(f">>> Directorio segmentado de entrada: {SEGMENTED_WAV_DIR}")
    print(f">>> Directorio de salida de embeddings: {EMBEDDINGS_OUTPUT_DIR}")
    print(f">>> Modelo wav2vec2: {MODEL_ID}")
    print(f">>> Dispositivo de inferencia: {DEVICE_NAME}")
    feature_extractor, model = load_model_and_extractor()
    feature_extractor_sampling_rate = int(getattr(feature_extractor, "sampling_rate", EMBEDDING_SAMPLE_RATE))
    if feature_extractor_sampling_rate != EMBEDDING_SAMPLE_RATE:
        raise ValueError(
            f"Sampling rate inconsistente: extractor={feature_extractor_sampling_rate}, "
            f"configurado={EMBEDDING_SAMPLE_RATE}"
        )

    min_input_samples = compute_min_input_samples(model)
    pass_attention_mask_to_model = use_attention_mask_for_model(model)

    split_summaries = {}
    all_errors = []
    for split_name in ("train", "validation", "test"):
        summary, errors = process_split(
            feature_extractor=feature_extractor,
            model=model,
            split_name=split_name,
            min_input_samples=min_input_samples,
            pass_attention_mask_to_model=pass_attention_mask_to_model,
        )
        split_summaries[split_name] = summary
        all_errors.extend(errors)
        print(
            f">>> {split_name}: {summary['n_segments']} segmentos | "
            f"normal={summary['n_negative']} | adventitious={summary['n_positive']} | "
            f"errors={summary['n_errors']}"
        )

    write_manifest(
        split_summaries=split_summaries,
        all_errors=all_errors,
        model=model,
        feature_extractor=feature_extractor,
        min_input_samples=min_input_samples,
        pass_attention_mask_to_model=pass_attention_mask_to_model,
    )
    if all_errors:
        print(f">>> Aviso: hubo {len(all_errors)} segmentos con error. Revisa embedding_manifest.json.")


if __name__ == "__main__":
    main()
