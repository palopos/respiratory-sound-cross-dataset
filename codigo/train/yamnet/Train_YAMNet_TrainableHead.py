from datetime import datetime
import json
import os
import random

# Locate Fix_Paper1 root to keep all paths relative to the project.
_SCRIPT_DIR = os.path.normpath(os.path.dirname(__file__) or ".")


def _find_project_root(start_dir):
    current_dir = os.path.abspath(start_dir)
    while True:
        if os.path.exists(os.path.join(current_dir, "environment.yml")) and os.path.isdir(os.path.join(current_dir, "ficheros")):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise FileNotFoundError("Could not locate the Fix_Paper1 project root.")
        current_dir = parent_dir


_PROJECT_ROOT = _find_project_root(_SCRIPT_DIR)
os.chdir(_PROJECT_ROOT)
_PROJECT_DIR = "."
_CACHE_DIR = os.path.join(_PROJECT_DIR, ".cache")
_MPLCONFIGDIR = os.path.join(_PROJECT_DIR, ".matplotlib")
_DEFAULT_CPU_THREAD_COUNT = str(min(8, os.cpu_count() or 4))
os.environ.setdefault("XDG_CACHE_HOME", _CACHE_DIR)
os.environ.setdefault("MPLCONFIGDIR", _MPLCONFIGDIR)
os.environ.setdefault("OMP_NUM_THREADS", _DEFAULT_CPU_THREAD_COUNT)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", os.environ["OMP_NUM_THREADS"])
os.environ.setdefault("NUMEXPR_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", os.environ["OMP_NUM_THREADS"])
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
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
)
from tqdm import tqdm

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
        raise ValueError(f"{env_name} does not contain any valid integer values.")
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
    return "".join(chars).strip("-") or "no-label"


def resolve_device_name():
    return "gpu" if tf.config.list_physical_devices("GPU") else "cpu"


DATASET_NAME = os.environ.get("DATASET_NAME", "0 SPRSound")
DATASET_LABEL = os.environ.get("DATASET_LABEL", "sprsound")
TRAIN_DATASET_LABEL = os.environ.get("TRAIN_DATASET_LABEL", DATASET_LABEL)
VALIDATION_DATASET_LABEL = os.environ.get("VALIDATION_DATASET_LABEL", TRAIN_DATASET_LABEL)
TEST_DATASET_LABEL = os.environ.get("TEST_DATASET_LABEL", DATASET_LABEL)
DATASET_PROTOCOL = os.environ.get(
    "DATASET_PROTOCOL",
    f"{slugify(TRAIN_DATASET_LABEL)}_to_{slugify(TEST_DATASET_LABEL)}",
)
EXPERIMENT_BLOCK = os.environ.get("EXPERIMENT_BLOCK", "04_yamnet_trainable_head")
MODEL_FAMILY = os.environ.get("MODEL_FAMILY", "yamnet")
MODEL_DISPLAY_NAME = os.environ.get("MODEL_DISPLAY_NAME", "YAMNet")
BACKBONE_NAME = os.environ.get("BACKBONE_NAME", "google_yamnet_tfhub")
TRAINING_MODE = os.environ.get("TRAINING_MODE", "frozen_backbone_trainable_head")
CLASSIFIER_TYPE = os.environ.get("CLASSIFIER_TYPE", "trainable_mlp")
EMBEDDING_MODE = os.environ.get("EMBEDDING_MODE", "yamnet_mean_pool_1024")
FINETUNING_SCOPE = os.environ.get("FINETUNING_SCOPE", "classification_head_only")
EXPERIMENT_NOTES = os.environ.get(
    "EXPERIMENT_NOTES",
    "Screening run with frozen YAMNet embeddings and a trainable classification head.",
)
CHECKPOINT_SELECTION_METRIC = os.environ.get("CHECKPOINT_SELECTION_METRIC", "validation_auc_roc")
SPLIT_SEED = int(os.environ.get("SPLIT_SEED", 42))
DEVICE_NAME = resolve_device_name()
SEEDS = parse_int_list_env("EXPERIMENT_SEEDS", [42])
SEED_LIST_LABEL = "-".join(str(seed) for seed in SEEDS)
TRAINING_SEEDS_TAG = f"seeds_{SEED_LIST_LABEL}"
REPORT_THRESHOLD_STRATEGY = os.environ.get("REPORT_THRESHOLD_STRATEGY", "val_best_f1").strip()

CLASS_WEIGHT_MODE = os.environ.get("CLASS_WEIGHT_MODE", "none").strip().lower()
WEIGHT1 = float(os.environ.get("CLASS_WEIGHT_NEGATIVE", 1.0))
WEIGHT2 = float(os.environ.get("CLASS_WEIGHT_POSITIVE", 1.0))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 128))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", 20))
EARLY_STOPPING_PATIENCE = int(os.environ.get("EARLY_STOPPING_PATIENCE", 4))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 1e-3))
HIDDEN_UNITS = parse_int_list_env("MLP_HIDDEN_UNITS", [256, 64])
DROPOUT_RATE = float(os.environ.get("DROPOUT_RATE", 0.30))
L2_REG = float(os.environ.get("L2_REG", 1e-4))
PREDICT_BATCH_SIZE = int(os.environ.get("PREDICT_BATCH_SIZE", 256))
MIXUP_ENABLED = os.environ.get("MIXUP_ENABLED", "0") == "1"
MIXUP_ALPHA = float(os.environ.get("MIXUP_ALPHA", 0.2))
MIXUP_MULTIPLIER = int(os.environ.get("MIXUP_MULTIPLIER", 1))

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

DEFAULT_EMBEDDINGS_BASE_DIR = os.path.join(
    _PROJECT_DIR,
    "ficheros",
    "3 Embeddings YAMNet",
    f"{DATASET_NAME}_seed{SPLIT_SEED}_yamnet_tfhub_mean_pool_1024",
)
EMBEDDINGS_BASE_DIR = os.environ.get("EMBEDDINGS_BASE_DIR", DEFAULT_EMBEDDINGS_BASE_DIR)
TRAIN_EMBEDDINGS_FILE = os.environ.get(
    "TRAIN_EMBEDDINGS_FILE",
    os.path.join(EMBEDDINGS_BASE_DIR, "segmentado_train_embeddings.npz"),
)
TRAIN_METADATA_FILE = os.environ.get(
    "TRAIN_METADATA_FILE",
    os.path.join(EMBEDDINGS_BASE_DIR, "segmentado_train_metadata.csv"),
)
VALIDATION_EMBEDDINGS_FILE = os.environ.get(
    "VALIDATION_EMBEDDINGS_FILE",
    os.path.join(EMBEDDINGS_BASE_DIR, "segmentado_validation_embeddings.npz"),
)
VALIDATION_METADATA_FILE = os.environ.get(
    "VALIDATION_METADATA_FILE",
    os.path.join(EMBEDDINGS_BASE_DIR, "segmentado_validation_metadata.csv"),
)
TEST_EMBEDDINGS_FILE = os.environ.get(
    "TEST_EMBEDDINGS_FILE",
    os.path.join(EMBEDDINGS_BASE_DIR, "segmentado_test_embeddings.npz"),
)
TEST_METADATA_FILE = os.environ.get(
    "TEST_METADATA_FILE",
    os.path.join(EMBEDDINGS_BASE_DIR, "segmentado_test_metadata.csv"),
)

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


def configure_tf_runtime():
    tf.get_logger().setLevel("ERROR")
    try:
        tf.config.threading.set_intra_op_parallelism_threads(int(os.environ["TF_NUM_INTRAOP_THREADS"]))
        tf.config.threading.set_inter_op_parallelism_threads(int(os.environ["TF_NUM_INTEROP_THREADS"]))
    except RuntimeError:
        pass


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
            "embeddings_by_split": {
                "train": {"npz": TRAIN_EMBEDDINGS_FILE, "metadata": TRAIN_METADATA_FILE},
                "validation": {"npz": VALIDATION_EMBEDDINGS_FILE, "metadata": VALIDATION_METADATA_FILE},
                "test": {"npz": TEST_EMBEDDINGS_FILE, "metadata": TEST_METADATA_FILE},
            },
        },
        "training": {
            "device": DEVICE_NAME,
            "batch_size": BATCH_SIZE,
            "predict_batch_size": PREDICT_BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "hidden_units": HIDDEN_UNITS,
            "dropout_rate": DROPOUT_RATE,
            "l2_reg": L2_REG,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "class_weight_mode": CLASS_WEIGHT_MODE,
            "weight1": WEIGHT1,
            "weight2": WEIGHT2,
            "checkpoint_selection_metric": CHECKPOINT_SELECTION_METRIC,
            "report_threshold_strategy": REPORT_THRESHOLD_STRATEGY,
            "training_seeds": SEEDS,
            "mixup_enabled": MIXUP_ENABLED,
            "mixup_alpha": MIXUP_ALPHA,
            "mixup_multiplier": MIXUP_MULTIPLIER,
        },
    }


def write_run_manifest():
    manifest_path = os.path.join(MODEL_OUTPUT_DIR, "run_manifest.json")
    write_json(manifest_path, build_run_manifest())
    print(f">>> YAMNet trainable-head run manifest saved to: {manifest_path}")


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


def assert_embeddings_ready():
    required = [
        TRAIN_EMBEDDINGS_FILE,
        TRAIN_METADATA_FILE,
        VALIDATION_EMBEDDINGS_FILE,
        VALIDATION_METADATA_FILE,
        TEST_EMBEDDINGS_FILE,
        TEST_METADATA_FILE,
    ]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Missing YAMNet embedding artifacts: "
            + ", ".join(missing)
            + ". Run the corresponding YAMNet frozen preprocessing first."
        )


def load_split_dataset(embeddings_file, metadata_file):
    payload = np.load(embeddings_file, allow_pickle=False)
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    labels = np.asarray(payload["labels"], dtype=np.int64)
    metadata = pd.read_csv(metadata_file)
    if len(embeddings) != len(labels) or len(embeddings) != len(metadata):
        raise ValueError(
            f"Mismatch in {embeddings_file} / {metadata_file}: "
            f"{len(embeddings)} embeddings, {len(labels)} labels, {len(metadata)} metadata rows"
        )
    return embeddings, labels, metadata


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def resolve_class_weights(train_labels):
    if CLASS_WEIGHT_MODE == "none":
        return None, {}
    if CLASS_WEIGHT_MODE == "manual":
        return {0: float(WEIGHT1), 1: float(WEIGHT2)}, {"negative": WEIGHT1, "positive": WEIGHT2}
    if CLASS_WEIGHT_MODE == "balanced":
        train_labels = np.asarray(train_labels, dtype=np.int64)
        total = float(len(train_labels))
        negatives = max(1.0, float(np.sum(train_labels == 0)))
        positives = max(1.0, float(np.sum(train_labels == 1)))
        negative_weight = total / (2.0 * negatives)
        positive_weight = total / (2.0 * positives)
        return (
            {0: float(negative_weight), 1: float(positive_weight)},
            {"negative": negative_weight, "positive": positive_weight},
        )
    raise ValueError(f"Unsupported CLASS_WEIGHT_MODE: {CLASS_WEIGHT_MODE}")


def build_model(train_x):
    normalizer = tf.keras.layers.Normalization(axis=-1, name="input_normalization")
    normalizer.adapt(train_x)

    inputs = tf.keras.Input(shape=(train_x.shape[1],), name="yamnet_embedding")
    x = normalizer(inputs)
    for layer_idx, units in enumerate(HIDDEN_UNITS, start=1):
        x = tf.keras.layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
            name=f"dense_{layer_idx}",
        )(x)
        x = tf.keras.layers.Dropout(DROPOUT_RATE if layer_idx == 1 else max(0.05, DROPOUT_RATE / 2.0), name=f"dropout_{layer_idx}")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="adventitious_probability")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="yamnet_trainable_head")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc_roc", curve="ROC"),
            tf.keras.metrics.AUC(name="auc_pr", curve="PR"),
        ],
    )
    return model


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
        eval_logloss = log_loss(y_true, probabilities, labels=[0, 1])
    else:
        roc_auc, pr_auc, eval_logloss = np.nan, np.nan, np.nan
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
        "loss": eval_logloss,
        "f1_adv": f1_score(y_true, predictions, zero_division=0),
        "recall_adv": recall_score(y_true, predictions, zero_division=0),
        "precision_adv": precision_score(y_true, predictions, zero_division=0),
        "threshold": threshold,
    }


def apply_decision_threshold(metrics, threshold, threshold_strategy):
    updated = compute_split_metrics(metrics["y_true"], metrics["probabilities"], threshold=threshold)
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


def export_split_report(split_name, split_metrics, metadata, excel_file, predictions_file, figure_file, seed=None, best_epoch=None):
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
        "Batch_Size": BATCH_SIZE,
        "Learning_Rate": LEARNING_RATE,
        "Best_Epoch": best_epoch,
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
        eval_logloss = log_loss(y_true, scores, labels=[0, 1])
    else:
        roc_auc, pr_auc, eval_logloss = np.nan, np.nan, np.nan

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
        "LogLoss": eval_logloss,
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


def history_row_key(row):
    val_auc_roc = float(row.get("val_auc_roc", np.nan))
    val_auc_pr = float(row.get("val_auc_pr", np.nan))
    val_loss = float(row.get("val_loss", np.nan))
    roc_key = val_auc_roc if np.isfinite(val_auc_roc) else -np.inf
    pr_key = val_auc_pr if np.isfinite(val_auc_pr) else -np.inf
    loss_key = -val_loss if np.isfinite(val_loss) else -np.inf
    return (roc_key, pr_key, loss_key)


def save_training_status(seed_dirs, payload):
    status_path = os.path.join(seed_dirs["history"], "training_status.json")
    write_json(status_path, payload)
    return status_path


class HistoryStatusCallback(tf.keras.callbacks.Callback):
    def __init__(self, seed, seed_dirs):
        super().__init__()
        self.seed = seed
        self.seed_dirs = seed_dirs
        self.rows = []
        self.history_file = os.path.join(seed_dirs["history"], "training_history.csv")

    def _best_row(self):
        if not self.rows:
            return None
        return max(self.rows, key=history_row_key)

    def _write_status(self, status, current_epoch):
        best_row = self._best_row()
        payload = {
            "status": status,
            "seed": self.seed,
            "current_epoch": int(current_epoch),
            "total_epochs": NUM_EPOCHS,
            "best_epoch": int(best_row["epoch"]) if best_row else 0,
            "best_validation_auc_roc": float(best_row.get("val_auc_roc", np.nan)) if best_row else np.nan,
            "best_validation_auc_pr": float(best_row.get("val_auc_pr", np.nan)) if best_row else np.nan,
            "history_file": self.history_file,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        save_training_status(self.seed_dirs, payload)

    def on_train_begin(self, logs=None):
        self._write_status("running", current_epoch=0)

    def on_epoch_begin(self, epoch, logs=None):
        print(f">>> Starting epoch {epoch + 1}/{NUM_EPOCHS} for seed {self.seed}...")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        row = {"epoch": int(epoch + 1)}
        for key, value in logs.items():
            if value is None:
                row[key] = np.nan
            else:
                row[key] = float(value)
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.history_file, index=False)
        self._write_status("running", current_epoch=epoch + 1)
        print(
            f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | "
            f"loss={row.get('loss', np.nan):.4f} | "
            f"acc={row.get('accuracy', np.nan):.4f} | "
            f"val_loss={row.get('val_loss', np.nan):.4f} | "
            f"val_auc_roc={row.get('val_auc_roc', np.nan):.4f} | "
            f"val_auc_pr={row.get('val_auc_pr', np.nan):.4f}"
        )

    def on_train_end(self, logs=None):
        current_epoch = self.rows[-1]["epoch"] if self.rows else 0
        self._write_status("completed_training_loop", current_epoch=current_epoch)


def evaluate_model(model, embeddings, labels, metadata):
    probabilities = clip_probabilities(
        model.predict(embeddings, batch_size=PREDICT_BATCH_SIZE, verbose=0).reshape(-1)
    )
    metrics = compute_split_metrics(labels, probabilities)
    metrics["metadata"] = metadata.copy()
    return metrics


def build_mixup_training_data(train_x, train_y, seed):
    train_x = np.asarray(train_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.float32)
    if not MIXUP_ENABLED or MIXUP_MULTIPLIER <= 0:
        return train_x, train_y, {"enabled": False}

    rng = np.random.default_rng(seed)
    n_samples = len(train_y)
    mixed_x_parts = [train_x]
    mixed_y_parts = [train_y]

    for _ in range(MIXUP_MULTIPLIER):
        index_a = rng.permutation(n_samples)
        index_b = rng.permutation(n_samples)
        lam = rng.beta(MIXUP_ALPHA, MIXUP_ALPHA, size=n_samples).astype(np.float32)
        lam_x = lam[:, None]
        mixed_x = lam_x * train_x[index_a] + (1.0 - lam_x) * train_x[index_b]
        mixed_y = lam * train_y[index_a] + (1.0 - lam) * train_y[index_b]
        mixed_x_parts.append(mixed_x.astype(np.float32))
        mixed_y_parts.append(mixed_y.astype(np.float32))

    augmented_x = np.concatenate(mixed_x_parts, axis=0).astype(np.float32)
    augmented_y = np.concatenate(mixed_y_parts, axis=0).astype(np.float32)
    return augmented_x, augmented_y, {
        "enabled": True,
        "alpha": MIXUP_ALPHA,
        "multiplier": MIXUP_MULTIPLIER,
        "original_train_rows": int(n_samples),
        "augmented_train_rows": int(len(augmented_y)),
    }


def train_single_seed(seed, train_x, train_y, validation_x, validation_y, validation_meta, test_x, test_y, test_meta):
    configure_tf_runtime()
    tf.keras.backend.clear_session()
    set_seed(seed)

    seed_dirs = build_seed_output_dirs(seed)
    class_weights, class_weight_summary = resolve_class_weights(train_y)
    fit_train_x, fit_train_y, mixup_summary = build_mixup_training_data(train_x, train_y, seed)
    write_seed_manifest(
        seed=seed,
        seed_dirs=seed_dirs,
        dataset_sizes={
            "train": len(train_y),
            "validation": len(validation_y),
            "test": len(test_y),
        },
        training_summary={
            "class_weights": class_weight_summary,
            "mixup": mixup_summary,
        },
    )

    print(
        f"\n--- YAMNet trainable head | Seed {seed} ---\n"
        f"train={len(train_y)} | validation={len(validation_y)} | test={len(test_y)}"
    )
    print(f">>> Device: {DEVICE_NAME}")

    model = build_model(train_x)
    best_model_path = os.path.join(seed_dirs["checkpoints"], "best_model.keras")
    callbacks = [
        HistoryStatusCallback(seed=seed, seed_dirs=seed_dirs),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_auc_roc",
            mode="max",
            save_best_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc_roc",
            mode="max",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=0,
        ),
    ]

    model.fit(
        fit_train_x,
        fit_train_y,
        validation_data=(validation_x, validation_y),
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0,
    )

    history_file = os.path.join(seed_dirs["history"], "training_history.csv")
    history_df = pd.read_csv(history_file) if os.path.exists(history_file) else pd.DataFrame()
    if history_df.empty:
        raise RuntimeError("Training finished without producing training history.")
    best_history_row = max(history_df.to_dict("records"), key=history_row_key)
    best_epoch = int(best_history_row["epoch"])

    best_model = tf.keras.models.load_model(best_model_path, compile=False)
    print(f">>> Best trainable-head checkpoint saved to: {best_model_path}")

    validation_raw = evaluate_model(best_model, validation_x, validation_y, validation_meta)
    threshold_strategy, report_threshold, available_thresholds = resolve_report_threshold(
        validation_raw["y_true"],
        validation_raw["probabilities"],
    )
    validation_metrics = apply_decision_threshold(validation_raw, report_threshold, threshold_strategy)

    test_raw = evaluate_model(best_model, test_x, test_y, test_meta)
    test_metrics = apply_decision_threshold(test_raw, report_threshold, threshold_strategy)

    excel_path = os.path.join(seed_dirs["reports"], "metricas.xlsx")
    validation_predictions_path = os.path.join(seed_dirs["predictions"], "predicciones_validation.csv")
    test_predictions_path = os.path.join(seed_dirs["predictions"], "predicciones_test.csv")
    validation_figure_path = os.path.join(seed_dirs["figures"], "analisis_validation.png")
    test_figure_path = os.path.join(seed_dirs["figures"], "analisis_test.png")

    validation_row = export_split_report(
        split_name="validation",
        split_metrics=validation_metrics,
        metadata=validation_metrics["metadata"],
        excel_file=excel_path,
        predictions_file=validation_predictions_path,
        figure_file=validation_figure_path,
        seed=seed,
        best_epoch=best_epoch,
    )
    test_row = export_split_report(
        split_name="test",
        split_metrics=test_metrics,
        metadata=test_metrics["metadata"],
        excel_file=excel_path,
        predictions_file=test_predictions_path,
        figure_file=test_figure_path,
        seed=seed,
        best_epoch=best_epoch,
    )

    threshold_df = analyze_thresholds_and_calibration(
        validation_predictions_file=validation_predictions_path,
        test_predictions_file=test_predictions_path,
        output_dir=seed_dirs["calibration"],
        seed=seed,
    )

    checkpoint_summary = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_auc_roc": float(best_history_row.get("val_auc_roc", np.nan)),
        "best_validation_auc_pr": float(best_history_row.get("val_auc_pr", np.nan)),
        "best_validation_loss": float(best_history_row.get("val_loss", np.nan)),
        "report_threshold_strategy": threshold_strategy,
        "report_threshold": report_threshold,
        "available_validation_thresholds": available_thresholds,
        "class_weight_summary": class_weight_summary,
        "model_path": best_model_path,
    }
    write_json(os.path.join(seed_dirs["checkpoints"], "best_checkpoint_summary.json"), checkpoint_summary)

    save_training_status(
        seed_dirs,
        {
            "status": "completed",
            "seed": seed,
            "current_epoch": int(history_df["epoch"].max()),
            "total_epochs": NUM_EPOCHS,
            "best_epoch": best_epoch,
            "best_validation_auc_roc": float(best_history_row.get("val_auc_roc", np.nan)),
            "best_validation_auc_pr": float(best_history_row.get("val_auc_pr", np.nan)),
            "history_file": history_file,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    write_seed_manifest(
        seed=seed,
        seed_dirs=seed_dirs,
        dataset_sizes={
            "train": len(train_y),
            "validation": len(validation_y),
            "test": len(test_y),
        },
        training_summary=checkpoint_summary,
    )
    tf.keras.backend.clear_session()
    return [validation_row, test_row], threshold_df


def run_training():
    configure_tf_runtime()
    assert_embeddings_ready()
    os.makedirs(SUMMARY_OUTPUT_DIR, exist_ok=True)
    write_run_manifest()
    print(f">>> Experiment root: {MODEL_OUTPUT_DIR}")
    print(f">>> Multi-seed summaries: {SUMMARY_OUTPUT_DIR}")
    print(f">>> Training seeds: {SEEDS}")
    print(f">>> Report threshold strategy: {REPORT_THRESHOLD_STRATEGY}")
    if len(SEEDS) == 1:
        print(">>> Screening mode: one seed by default. Promote only promising protocols to multi-seed confirmation.")

    train_x, train_y, _ = load_split_dataset(TRAIN_EMBEDDINGS_FILE, TRAIN_METADATA_FILE)
    validation_x, validation_y, validation_meta = load_split_dataset(VALIDATION_EMBEDDINGS_FILE, VALIDATION_METADATA_FILE)
    test_x, test_y, test_meta = load_split_dataset(TEST_EMBEDDINGS_FILE, TEST_METADATA_FILE)

    final_rows = []
    threshold_rows = []
    for seed in progress(SEEDS, desc="Seeds YAMNet trainable head", unit="seed"):
        split_rows, threshold_df = train_single_seed(
            seed=seed,
            train_x=train_x,
            train_y=train_y,
            validation_x=validation_x,
            validation_y=validation_y,
            validation_meta=validation_meta,
            test_x=test_x,
            test_y=test_y,
            test_meta=test_meta,
        )
        final_rows.extend(split_rows)
        threshold_rows.append(threshold_df)

    df_results = pd.DataFrame(final_rows)
    summary_path = os.path.join(SUMMARY_OUTPUT_DIR, "resumen_seeds.csv")
    df_results.to_csv(summary_path, index=False)
    print(f">>> Per-seed summary saved to: {summary_path}")

    if threshold_rows:
        df_thresholds = pd.concat(threshold_rows, ignore_index=True)
        thresholds_summary_path = os.path.join(SUMMARY_OUTPUT_DIR, "threshold_calibration_resumen_seeds.csv")
        df_thresholds.to_csv(thresholds_summary_path, index=False)
        print(f">>> Threshold/calibration summary saved to: {thresholds_summary_path}")
    else:
        df_thresholds = None

    if len(SEEDS) > 1:
        metric_cols = ["AUC_ROC", "AUC_PR", "F1-Score_Adv", "Recall_Adv", "Precision_Adv", "TN", "FP", "FN", "TP"]
        resumen_media_std_path = os.path.join(SUMMARY_OUTPUT_DIR, "media_std_por_split.csv")
        df_results.groupby("Split")[metric_cols].agg(["mean", "std"]).to_csv(resumen_media_std_path)
        print(f">>> Mean/std by split saved to: {resumen_media_std_path}")

        if df_thresholds is not None:
            threshold_metric_cols = ["Accuracy", "AUC_ROC", "AUC_PR", "Brier", "LogLoss", "F1_Adv", "Recall_Adv", "Specificity", "Precision_Adv"]
            thresholds_media_std_path = os.path.join(SUMMARY_OUTPUT_DIR, "threshold_calibration_media_std.csv")
            df_thresholds.groupby(["Calibration", "Threshold_Strategy", "Split"])[threshold_metric_cols].agg(["mean", "std"]).to_csv(thresholds_media_std_path)
            print(f">>> Threshold/calibration mean/std saved to: {thresholds_media_std_path}")


def main():
    run_training()


if __name__ == "__main__":
    main()
