from datetime import datetime
import json
import os

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
_MPLCONFIGDIR = os.path.join(_PROJECT_DIR, ".matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", _CACHE_DIR)
os.environ.setdefault("MPLCONFIGDIR", _MPLCONFIGDIR)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump, load
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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


DATASET_NAME = os.environ.get("DATASET_NAME", "0 SPRSound")
DATASET_LABEL = os.environ.get("DATASET_LABEL", "sprsound")
TRAIN_DATASET_LABEL = os.environ.get("TRAIN_DATASET_LABEL", DATASET_LABEL)
VALIDATION_DATASET_LABEL = os.environ.get("VALIDATION_DATASET_LABEL", TRAIN_DATASET_LABEL)
TEST_DATASET_LABEL = os.environ.get("TEST_DATASET_LABEL", DATASET_LABEL)
DATASET_PROTOCOL = os.environ.get(
    "DATASET_PROTOCOL",
    f"{slugify(TRAIN_DATASET_LABEL)}_to_{slugify(TEST_DATASET_LABEL)}",
)
EXPERIMENT_BLOCK = os.environ.get("EXPERIMENT_BLOCK", "02_wav2vec2_frozen_lr")
MODEL_FAMILY = os.environ.get("MODEL_FAMILY", "wav2vec2")
MODEL_DISPLAY_NAME = os.environ.get("MODEL_DISPLAY_NAME", "Wav2Vec2")
BACKBONE_NAME = os.environ.get("BACKBONE_NAME", "facebook_wav2vec2_base")
TRAINING_MODE = os.environ.get("TRAINING_MODE", "frozen")
CLASSIFIER_TYPE = os.environ.get("CLASSIFIER_TYPE", "logistic_regression")
EMBEDDING_MODE = os.environ.get("EMBEDDING_MODE", "wav2vec2_mean_pool_768")
FINETUNING_SCOPE = os.environ.get("FINETUNING_SCOPE", "none")
EXPERIMENT_NOTES = os.environ.get("EXPERIMENT_NOTES", "")
CHECKPOINT_SELECTION_METRIC = os.environ.get("CHECKPOINT_SELECTION_METRIC", "not_applicable_frozen_lr")
SPLIT_SEED = int(os.environ.get("SPLIT_SEED", 42))
DETERMINISTIC_BASELINE = True
SEEDS = parse_int_list_env("EXPERIMENT_SEEDS", [42])
SEED_LIST_LABEL = "-".join(str(seed) for seed in SEEDS)
TRAINING_SEEDS_TAG = f"seeds_{SEED_LIST_LABEL}"

CLASS_WEIGHT_MODE = os.environ.get("CLASS_WEIGHT_MODE", "none").strip().lower()
WEIGHT1 = float(os.environ.get("CLASS_WEIGHT_NEGATIVE", 1.0))
WEIGHT2 = float(os.environ.get("CLASS_WEIGHT_POSITIVE", 1.0))
LOGREG_C = float(os.environ.get("LOGREG_C", 1.0))
LOGREG_SOLVER = os.environ.get("LOGREG_SOLVER", "liblinear")
MAX_ITER = int(os.environ.get("MAX_ITER", 2000))

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

DEFAULT_EMBEDDINGS_BASE_DIR = os.path.join(
    _PROJECT_DIR,
    "ficheros",
    "3 Embeddings wav2vec2",
    f"{DATASET_NAME}_seed{SPLIT_SEED}_wav2vec2_hf_mean_pool_768",
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
            "embeddings_base_dir": EMBEDDINGS_BASE_DIR,
            "default_embeddings_base_dir": DEFAULT_EMBEDDINGS_BASE_DIR,
            "train_embeddings_file": TRAIN_EMBEDDINGS_FILE,
            "train_metadata_file": TRAIN_METADATA_FILE,
            "validation_embeddings_file": VALIDATION_EMBEDDINGS_FILE,
            "validation_metadata_file": VALIDATION_METADATA_FILE,
            "test_embeddings_file": TEST_EMBEDDINGS_FILE,
            "test_metadata_file": TEST_METADATA_FILE,
        },
        "training": {
            "deterministic_baseline": DETERMINISTIC_BASELINE,
            "class_weight_mode": CLASS_WEIGHT_MODE,
            "weight1": WEIGHT1,
            "weight2": WEIGHT2,
            "logreg_c": LOGREG_C,
            "logreg_solver": LOGREG_SOLVER,
            "max_iter": MAX_ITER,
            "checkpoint_selection_metric": CHECKPOINT_SELECTION_METRIC,
            "training_seeds": SEEDS,
        },
    }


def write_run_manifest():
    manifest_path = os.path.join(MODEL_OUTPUT_DIR, "run_manifest.json")
    write_json(manifest_path, build_run_manifest())
    print(f">>> Manifiesto del experimento wav2vec2 frozen guardado en: {manifest_path}")


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


def assert_embeddings_ready():
    required = [
        TRAIN_EMBEDDINGS_FILE, TRAIN_METADATA_FILE,
        VALIDATION_EMBEDDINGS_FILE, VALIDATION_METADATA_FILE,
        TEST_EMBEDDINGS_FILE, TEST_METADATA_FILE,
    ]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Faltan artefactos de embeddings wav2vec2: "
            + ", ".join(missing)
            + ". Ejecuta antes el preprocesado Wav2Vec2 correspondiente."
        )


def load_split_dataset(embeddings_file, metadata_file):
    payload = np.load(embeddings_file, allow_pickle=False)
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    labels = np.asarray(payload["labels"], dtype=np.int64)
    metadata = pd.read_csv(metadata_file)
    if len(embeddings) != len(labels) or len(embeddings) != len(metadata):
        raise ValueError(
            f"Desajuste en {embeddings_file} / {metadata_file}: "
            f"{len(embeddings)} embeddings, {len(labels)} labels, {len(metadata)} filas metadata"
        )
    return embeddings, labels, metadata


def resolve_class_weight():
    if CLASS_WEIGHT_MODE == "none":
        return None
    if CLASS_WEIGHT_MODE == "balanced":
        return "balanced"
    if CLASS_WEIGHT_MODE == "manual":
        return {0: WEIGHT1, 1: WEIGHT2}
    raise ValueError(f"CLASS_WEIGHT_MODE no soportado: {CLASS_WEIGHT_MODE}")


def build_classifier(seed):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=LOGREG_C,
            solver=LOGREG_SOLVER,
            max_iter=MAX_ITER,
            random_state=seed,
            class_weight=resolve_class_weight(),
        )),
    ])


def clip_probabilities(probabilities, eps=1e-6):
    return np.clip(np.asarray(probabilities, dtype=np.float64), eps, 1.0 - eps)


def probability_to_logit(probabilities):
    probabilities = clip_probabilities(probabilities)
    return np.log(probabilities / (1.0 - probabilities))


def sigmoid_np(values):
    values = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-values))


def save_analysis_figure(y_true, y_pred, probabilities, figure_file):
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    plt.subplot(1, 3, 2)
    if len(fpr):
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.legend(loc="lower right")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.subplot(1, 3, 3)
    if len(precision_curve):
        plt.plot(recall_curve, precision_curve, color='darkgreen', lw=2, label=f'PR AUC = {pr_auc:.2f}')
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


def metricas_informes(model_path, split_name, embeddings, labels, metadata, excel_file, predictions_file, figure_file, seed=None):
    print("\n" + "=" * 30)
    print(f"GENERATING FINAL REPORT: {split_name.upper()}")
    print("=" * 30)

    model = load(model_path)
    probabilities = clip_probabilities(model.predict_proba(embeddings)[:, 1])
    predictions = (probabilities >= 0.5).astype(int)
    target_names = ["Healthy (Normal)", "Adventitious"]

    print(classification_report(labels, predictions, labels=[0, 1], target_names=target_names, zero_division=0))
    report_dict = classification_report(
        labels, predictions, labels=[0, 1], target_names=target_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    if len(np.unique(labels)) == 2:
        fpr, tpr, _ = roc_curve(labels, probabilities)
        roc_auc = auc(fpr, tpr)
        pr_auc = average_precision_score(labels, probabilities)
    else:
        roc_auc, pr_auc = np.nan, np.nan

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
    df_predictions["y_true"] = labels
    df_predictions["prob_adventitious"] = probabilities
    df_predictions["y_pred_threshold_0_5"] = predictions
    df_predictions["threshold"] = 0.5
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
        "Checkpoint_Selection": CHECKPOINT_SELECTION_METRIC,
        "Class_Weight_Mode": CLASS_WEIGHT_MODE,
        "Weight1": WEIGHT1,
        "Weight2": WEIGHT2,
        "LogReg_C": LOGREG_C,
        "LogReg_Solver": LOGREG_SOLVER,
        "Max_Iter": MAX_ITER,
        "AUC_ROC": round(roc_auc, 4) if not np.isnan(roc_auc) else np.nan,
        "AUC_PR": round(pr_auc, 4) if not np.isnan(pr_auc) else np.nan,
        "F1-Score_Adv": round(report_dict["Adventitious"]["f1-score"], 4),
        "Recall_Adv": round(report_dict["Adventitious"]["recall"], 4),
        "Precision_Adv": round(report_dict["Adventitious"]["precision"], 4),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }
    print(nuevo_resultado)

    df_nuevo = pd.DataFrame([nuevo_resultado])
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
        start_row_nuevo = start_row_cm + len(df_cm) + gap
        df_nuevo.to_excel(writer, sheet_name=sheet_name, startrow=start_row_nuevo)
    print(f">>> Informe exportado exitosamente a: {excel_file}")

    save_analysis_figure(labels, predictions, probabilities, figure_file)
    print(f">>> Figura de análisis guardada en: {figure_file}")

    return nuevo_resultado


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


def analizar_thresholds_y_calibracion(validation_predictions_file, test_predictions_file, output_dir, seed):
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
    metrics_file = os.path.join(output_dir, "threshold_calibration_metrics.csv")
    validation_calibrated_file = os.path.join(output_dir, "predicciones_validation_calibradas.csv")
    test_calibrated_file = os.path.join(output_dir, "predicciones_test_calibradas.csv")
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(metrics_file, index=False)
    validation_enriched.to_csv(validation_calibrated_file, index=False)
    test_enriched.to_csv(test_calibrated_file, index=False)
    print(f">>> Métricas de thresholds/calibración guardadas en: {metrics_file}")
    return df_metrics


def run_training():
    assert_embeddings_ready()
    os.makedirs(SUMMARY_OUTPUT_DIR, exist_ok=True)
    write_run_manifest()
    print(f">>> Carpeta raíz del experimento: {MODEL_OUTPUT_DIR}")
    print(f">>> Resúmenes multi-seed: {SUMMARY_OUTPUT_DIR}")
    print(f">>> Seeds de entrenamiento: {SEEDS}")
    if DETERMINISTIC_BASELINE:
        print(">>> Aviso: wav2vec2 frozen + logistic regression es un baseline practicamente determinista.")
        if len(SEEDS) > 1:
            print(">>> Varias seeds solo tienen sentido si quieres confirmar que el resultado no cambia.")

    train_x, train_y, _ = load_split_dataset(TRAIN_EMBEDDINGS_FILE, TRAIN_METADATA_FILE)
    validation_x, validation_y, validation_meta = load_split_dataset(VALIDATION_EMBEDDINGS_FILE, VALIDATION_METADATA_FILE)
    test_x, test_y, test_meta = load_split_dataset(TEST_EMBEDDINGS_FILE, TEST_METADATA_FILE)

    resultados_finales = []
    resultados_threshold_calibracion = []
    for seed in progress(SEEDS, desc="Seeds wav2vec2 frozen", unit="seed"):
        seed_dirs = build_seed_output_dirs(seed)
        write_seed_manifest(
            seed=seed,
            seed_dirs=seed_dirs,
            dataset_sizes={
                "train": len(train_y),
                "validation": len(validation_y),
                "test": len(test_y),
            },
        )
        print(
            f"\n--- wav2vec2 frozen | Seed {seed} ---\n"
            f"train={len(train_y)} | validation={len(validation_y)} | test={len(test_y)}"
        )
        model = build_classifier(seed)
        model.fit(train_x, train_y)
        model_path = os.path.join(seed_dirs["checkpoints"], "frozen_lr.joblib")
        dump(model, model_path)
        print(f">>> Modelo frozen LR guardado en: {model_path}")

        excel_path = os.path.join(seed_dirs["reports"], "metricas.xlsx")
        validation_predictions_path = os.path.join(seed_dirs["predictions"], "predicciones_validation.csv")
        test_predictions_path = os.path.join(seed_dirs["predictions"], "predicciones_test.csv")
        validation_figure_path = os.path.join(seed_dirs["figures"], "analisis_validation.png")
        test_figure_path = os.path.join(seed_dirs["figures"], "analisis_test.png")

        resultados_finales.append(metricas_informes(
            model_path=model_path,
            split_name="validation",
            embeddings=validation_x,
            labels=validation_y,
            metadata=validation_meta,
            excel_file=excel_path,
            predictions_file=validation_predictions_path,
            figure_file=validation_figure_path,
            seed=seed,
        ))
        resultados_finales.append(metricas_informes(
            model_path=model_path,
            split_name="test",
            embeddings=test_x,
            labels=test_y,
            metadata=test_meta,
            excel_file=excel_path,
            predictions_file=test_predictions_path,
            figure_file=test_figure_path,
            seed=seed,
        ))
        resultados_threshold_calibracion.append(analizar_thresholds_y_calibracion(
            validation_predictions_file=validation_predictions_path,
            test_predictions_file=test_predictions_path,
            output_dir=seed_dirs["calibration"],
            seed=seed,
        ))

    df_resultados = pd.DataFrame(resultados_finales)
    resumen_path = os.path.join(SUMMARY_OUTPUT_DIR, "resumen_seeds.csv")
    df_resultados.to_csv(resumen_path, index=False)
    print(f">>> Resumen por seed guardado en: {resumen_path}")

    if resultados_threshold_calibracion:
        df_thresholds = pd.concat(resultados_threshold_calibracion, ignore_index=True)
        thresholds_resumen_path = os.path.join(SUMMARY_OUTPUT_DIR, "threshold_calibration_resumen_seeds.csv")
        df_thresholds.to_csv(thresholds_resumen_path, index=False)
        print(f">>> Resumen thresholds/calibración por seed guardado en: {thresholds_resumen_path}")
    else:
        df_thresholds = None

    if len(SEEDS) > 1:
        metric_cols = ["AUC_ROC", "AUC_PR", "F1-Score_Adv", "Recall_Adv", "Precision_Adv", "TN", "FP", "FN", "TP"]
        resumen_media_std_path = os.path.join(SUMMARY_OUTPUT_DIR, "media_std_por_split.csv")
        df_resultados.groupby("Split")[metric_cols].agg(["mean", "std"]).to_csv(resumen_media_std_path)
        print(f">>> Media y desviación estándar por split guardadas en: {resumen_media_std_path}")

        if df_thresholds is not None:
            threshold_metric_cols = ["Accuracy", "AUC_ROC", "AUC_PR", "Brier", "LogLoss", "F1_Adv", "Recall_Adv", "Specificity", "Precision_Adv"]
            thresholds_media_std_path = os.path.join(SUMMARY_OUTPUT_DIR, "threshold_calibration_media_std.csv")
            df_thresholds.groupby(["Calibration", "Threshold_Strategy", "Split"])[threshold_metric_cols].agg(["mean", "std"]).to_csv(thresholds_media_std_path)
            print(f">>> Media/std thresholds-calibración guardadas en: {thresholds_media_std_path}")


def main():
    run_training()


if __name__ == "__main__":
    main()
