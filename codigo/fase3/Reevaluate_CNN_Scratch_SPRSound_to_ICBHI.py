from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def find_project_root(start_dir: Path) -> Path:
    current = start_dir.resolve()
    while True:
        if (current / "environment.yml").exists() and (current / "ficheros").is_dir():
            return current
        if current.parent == current:
            raise FileNotFoundError("Could not locate the Fix_Paper1 project root.")
        current = current.parent


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(SCRIPT_DIR)
os.chdir(PROJECT_ROOT)

CNN_DIR = PROJECT_ROOT / "codigo" / "train" / "cnn"
sys.path.insert(0, str(CNN_DIR))

SPRSOUND_SPECTROGRAM_DIR = "./ficheros/2 Espectrogramas/0 SPRSound_seed42_resize_dinamico_MEL64_SIZE64"
ICBHI_SPECTROGRAM_DIR = "./ficheros/2 Espectrogramas/0 ICBHI_seed42_resize_dinamico_MEL64_SIZE64"
SPRSOUND_SPLIT_JSON = "./ficheros/splits/0 SPRSound/splits_patient_independent_seed42.json"
SPRSOUND_SPLIT_CSV = "./ficheros/splits/0 SPRSound/splits_patient_independent_seed42.csv"
ICBHI_SPLIT_JSON = "./ficheros/splits/0 ICBHI/splits_patient_independent_seed42.json"
ICBHI_SPLIT_CSV = "./ficheros/splits/0 ICBHI/splits_patient_independent_seed42.csv"

DEFAULT_SOURCE_RUN = (
    "./modelos/00-cnn-scratch-baseline/sprsound-to-icbhi/cnn/"
    "cnn_biocas-resnext-attention_scratch_trainable-mlp/split_seed_42/"
    "run_20260521_132019"
)
RUN_ID = os.environ.get(
    "CORRECTED_RUN_ID",
    f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_corrected-full-icbhi-test",
)

os.environ["RUN_ID"] = RUN_ID
os.environ["EXPERIMENT_SEEDS"] = "42,43,44"
os.environ.setdefault("DATALOADER_NUM_WORKERS", "0")
os.environ["DATASET_PROTOCOL"] = "sprsound_to_icbhi"
os.environ["DATASET_LABEL"] = "sprsound"
os.environ["TRAIN_DATASET_LABEL"] = "sprsound"
os.environ["VALIDATION_DATASET_LABEL"] = "sprsound"
os.environ["TEST_DATASET_LABEL"] = "icbhi"
os.environ["EXPERIMENT_NOTES"] = (
    "Evaluation-only correction of the original SPRSound-to-ICBHI CNN run. "
    "The original validation-selected checkpoints are reused; all 1,414 ICBHI "
    "test events are evaluated, including the 435 Crackle-only events previously omitted."
)
os.environ["SPLIT_MANIFEST_JSON"] = SPRSOUND_SPLIT_JSON
os.environ["SPLIT_MANIFEST_CSV"] = SPRSOUND_SPLIT_CSV
os.environ["TRAIN_SPLIT_MANIFEST_JSON"] = SPRSOUND_SPLIT_JSON
os.environ["TRAIN_SPLIT_MANIFEST_CSV"] = SPRSOUND_SPLIT_CSV
os.environ["VALIDATION_SPLIT_MANIFEST_JSON"] = SPRSOUND_SPLIT_JSON
os.environ["VALIDATION_SPLIT_MANIFEST_CSV"] = SPRSOUND_SPLIT_CSV
os.environ["TEST_SPLIT_MANIFEST_JSON"] = ICBHI_SPLIT_JSON
os.environ["TEST_SPLIT_MANIFEST_CSV"] = ICBHI_SPLIT_CSV
os.environ["PATH_SALIDA_NPY"] = SPRSOUND_SPECTROGRAM_DIR
os.environ["TRAIN_NPY_PATH"] = f"{SPRSOUND_SPECTROGRAM_DIR}/npy/segmentado_train"
os.environ["VALIDATION_NPY_PATH"] = f"{SPRSOUND_SPECTROGRAM_DIR}/npy/segmentado_validation"
os.environ["TEST_NPY_PATH"] = f"{ICBHI_SPECTROGRAM_DIR}/npy/segmentado_test"

import pandas as pd

import Train_CNN_Scratch_SPRSound as pipeline


SEEDS = (42, 43, 44)
EXPECTED_TEST_CLASS_COUNTS = {
    "Crackle": 435,
    "Normal": 629,
    "Wheeze": 242,
    "Wheeze_Crackle": 108,
}
EXPECTED_TEST_EVENTS = sum(EXPECTED_TEST_CLASS_COUNTS.values())
SOURCE_RUN = Path(os.environ.get("SOURCE_RUN_DIR", DEFAULT_SOURCE_RUN)).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_relative(path: Path | str) -> str:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = (PROJECT_ROOT / resolved).resolve()
    return str(resolved.relative_to(PROJECT_ROOT))


def direct_class_counts(root_dir: Path) -> dict[str, int]:
    return {
        class_dir.name: sum(1 for path in class_dir.iterdir() if path.suffix == ".npy")
        for class_dir in sorted(root_dir.iterdir())
        if class_dir.is_dir()
    }


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class PreloadedNumpyDataset(pipeline.Dataset):
    def __init__(self, samples: list[tuple[str, int]]):
        self.items = []
        for path, label in pipeline.progress(
            samples,
            desc="Loading recovered Crackle events",
            unit="event",
        ):
            mel = pipeline.np.load(path)
            tensor = pipeline.torch.from_numpy(mel).unsqueeze(0).float()
            self.items.append((tensor, pipeline.torch.tensor(label, dtype=pipeline.torch.long), path))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]


def normalize_prediction_frame(frame: pd.DataFrame, seed: int, split_name: str) -> pd.DataFrame:
    required = {"sample_path", "y_true", "prob_adventitious"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")

    result = frame.copy()
    result["run_id"] = pipeline.RUN_ID
    result["experiment_block"] = pipeline.EXPERIMENT_BLOCK
    result["dataset_protocol"] = pipeline.DATASET_PROTOCOL
    result["model_family"] = pipeline.MODEL_FAMILY
    result["training_mode"] = pipeline.TRAINING_MODE
    result["classifier_type"] = pipeline.CLASSIFIER_TYPE
    result["seed"] = seed
    result["split"] = split_name
    result["y_true"] = result["y_true"].astype(int)
    result["prob_adventitious"] = result["prob_adventitious"].astype(float)
    result["y_pred_threshold_0_5"] = (result["prob_adventitious"] >= 0.5).astype(int)
    result["threshold"] = 0.5
    columns = [
        "run_id",
        "experiment_block",
        "dataset_protocol",
        "model_family",
        "training_mode",
        "classifier_type",
        "seed",
        "split",
        "sample_path",
        "y_true",
        "prob_adventitious",
        "y_pred_threshold_0_5",
        "threshold",
    ]
    return result[columns].sort_values("sample_path").reset_index(drop=True)


def infer_predictions(
    checkpoint: Path,
    dataset: PreloadedNumpyDataset,
    seed: int,
) -> pd.DataFrame:
    loader = pipeline.make_dataloader(
        dataset,
        batch_size=int(os.environ.get("CORRECTION_EVAL_BATCH_SIZE", "128")),
        shuffle=False,
    )
    model = pipeline.BioCAS_Advanced_CNN().to(pipeline.device)
    model.load_state_dict(pipeline.torch.load(checkpoint, map_location=pipeline.device))
    model.eval()

    paths: list[str] = []
    labels: list[int] = []
    probabilities: list[float] = []
    with pipeline.torch.no_grad():
        for images, batch_labels, batch_paths in pipeline.progress(
            loader,
            desc=f"Recovered Crackle inference seed {seed}",
            unit="batch",
            leave=False,
        ):
            outputs = model(images.to(pipeline.device))
            probs = pipeline.torch.softmax(outputs, dim=1)[:, 1]
            paths.extend(batch_paths)
            labels.extend(batch_labels.numpy().astype(int).tolist())
            probabilities.extend(probs.cpu().numpy().astype(float).tolist())

    return normalize_prediction_frame(
        pd.DataFrame(
            {
                "sample_path": paths,
                "y_true": labels,
                "prob_adventitious": probabilities,
            }
        ),
        seed=seed,
        split_name="test",
    )


def report_from_predictions(
    predictions: pd.DataFrame,
    split_name: str,
    excel_file: Path,
    predictions_file: Path,
    figure_file: Path,
    seed: int,
) -> dict:
    frame = normalize_prediction_frame(predictions, seed=seed, split_name=split_name)
    labels = frame["y_true"].to_numpy(dtype=int)
    probabilities = frame["prob_adventitious"].to_numpy(dtype=float)
    predicted = frame["y_pred_threshold_0_5"].to_numpy(dtype=int)
    target_names = ["Healthy (Normal)", "Adventitious"]

    report_dict = pipeline.classification_report(
        labels,
        predicted,
        labels=[0, 1],
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    print("\n" + "=" * 30)
    print(f"GENERATING FINAL REPORT: {split_name.upper()}")
    print("=" * 30)
    print(
        pipeline.classification_report(
            labels,
            predicted,
            labels=[0, 1],
            target_names=target_names,
            zero_division=0,
        )
    )

    cm = pipeline.confusion_matrix(labels, predicted, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr, tpr, _ = pipeline.roc_curve(labels, probabilities)
    roc_auc = pipeline.auc(fpr, tpr)
    precision_curve, recall_curve, _ = pipeline.precision_recall_curve(labels, probabilities)
    pr_auc = pipeline.average_precision_score(labels, probabilities)

    predictions_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(predictions_file, index=False)

    result = {
        "Fecha": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        **pipeline.EXPERIMENT_METADATA,
        "Seed": seed,
        "Split_Seed": pipeline.SPLIT_SEED,
        "Training_Seeds": pipeline.SEED_LIST_LABEL,
        "Split": split_name,
        "Modelo": pipeline.MODEL_DISPLAY_NAME,
        "Mels": pipeline.N_MELS,
        "Size": pipeline.SIZE,
        "Batch_Size": pipeline.BATCH_SIZE,
        "LR": pipeline.LEARNING_RATE,
        "Checkpoint_Selection": pipeline.CHECKPOINT_SELECTION_METRIC,
        "Weight1": pipeline.WEIGHT1,
        "Weight2": pipeline.WEIGHT2,
        "Dropout": pipeline.DROPOUT,
        "Epochs": pipeline.EPOCHS,
        "Groups": pipeline.GRUPOS,
        "AUC_ROC": round(float(roc_auc), 4),
        "AUC_PR": round(float(pr_auc), 4),
        "F1-Score_Adv": round(report_dict["Adventitious"]["f1-score"], 4),
        "Recall_Adv": round(report_dict["Adventitious"]["recall"], 4),
        "Precision_Adv": round(report_dict["Adventitious"]["precision"], 4),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    report_df = pd.DataFrame(report_dict).transpose()
    matrix_df = pd.DataFrame(
        cm,
        index=["True_Healthy", "True_Adventitious"],
        columns=["Pred_Healthy", "Pred_Adventitious"],
    )
    result_df = pd.DataFrame([result])
    sheet_name = split_name[:31]
    excel_file.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if excel_file.exists() else "w"
    writer_kwargs = {"engine": "openpyxl", "mode": mode}
    if mode == "a":
        writer_kwargs["if_sheet_exists"] = "replace"
    with pd.ExcelWriter(excel_file, **writer_kwargs) as writer:
        report_df.to_excel(writer, sheet_name=sheet_name)
        matrix_start = len(report_df) + 3
        matrix_df.to_excel(writer, sheet_name=sheet_name, startrow=matrix_start)
        result_df.to_excel(writer, sheet_name=sheet_name, startrow=matrix_start + len(matrix_df) + 3)

    fig = pipeline.plt.figure(figsize=(15, 5))
    axis_cm = fig.add_subplot(1, 3, 1)
    pipeline.sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=axis_cm,
    )
    axis_cm.set_title("Confusion Matrix")
    axis_cm.set_xlabel("Predicted label")
    axis_cm.set_ylabel("True label")

    axis_roc = fig.add_subplot(1, 3, 2)
    axis_roc.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC AUC = {roc_auc:.2f}")
    axis_roc.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axis_roc.set_xlim(0.0, 1.0)
    axis_roc.set_ylim(0.0, 1.05)
    axis_roc.set_xlabel("False Positive Rate (1 - Specificity)")
    axis_roc.set_ylabel("True Positive Rate (Sensitivity)")
    axis_roc.set_title("ROC Curve")
    axis_roc.legend(loc="lower right")

    axis_pr = fig.add_subplot(1, 3, 3)
    axis_pr.plot(
        recall_curve,
        precision_curve,
        color="darkgreen",
        lw=2,
        label=f"PR AUC = {pr_auc:.2f}",
    )
    axis_pr.set_xlim(0.0, 1.0)
    axis_pr.set_ylim(0.0, 1.05)
    axis_pr.set_xlabel("Recall")
    axis_pr.set_ylabel("Precision")
    axis_pr.set_title("Precision-Recall Curve")
    axis_pr.legend(loc="lower left")
    fig.tight_layout()
    figure_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_file, dpi=300)
    pipeline.plt.close(fig)
    print(result)
    return result


def write_correction_run_manifest(validation_events: int, test_counts: dict[str, int]) -> None:
    manifest = pipeline.build_run_manifest(target_width=pipeline.infer_target_width_from_preprocessed_data())
    manifest["artifact_type"] = "corrected_evaluation_only"
    manifest["evaluation_correction"] = {
        "reason": "The SPRSound loader omitted ICBHI's generic Crackle directory.",
        "source_run": project_relative(SOURCE_RUN),
        "training_reused": True,
        "checkpoint_selection_reused": "best validation PR-AUC from the original run",
        "validation_events": validation_events,
        "original_test_events": 979,
        "corrected_test_events": sum(test_counts.values()),
        "recovered_test_events": sum(test_counts.values()) - 979,
        "test_class_counts": test_counts,
        "completion_method": (
            "Reuse the original predictions for the 979 previously evaluated events and infer "
            "the 435 omitted Crackle events with the same stored checkpoint."
        ),
    }
    pipeline.write_json(str(Path(pipeline.MODEL_OUTPUT_DIR) / "run_manifest.json"), manifest)


def write_seed_manifest(
    seed: int,
    seed_dirs: dict[str, str],
    source_checkpoint: Path,
    copied_checkpoint: Path,
    train_events: int,
    validation_events: int,
    test_events: int,
) -> None:
    manifest = pipeline.build_run_manifest(target_width=pipeline.infer_target_width_from_preprocessed_data())
    manifest.update(
        {
            "artifact_type": "corrected_evaluation_only",
            "seed": seed,
            "seed_output_dirs": seed_dirs,
            "dataset_sizes": {
                "train": train_events,
                "validation": validation_events,
                "test": test_events,
            },
            "evaluation_correction": {
                "training_reused": True,
                "source_checkpoint": project_relative(source_checkpoint),
                "source_checkpoint_sha256": sha256(source_checkpoint),
                "copied_checkpoint": project_relative(copied_checkpoint),
                "original_test_events": 979,
                "corrected_test_events": test_events,
                "recovered_label": "Crackle",
                "recovered_events": 435,
                "completion_method": "original 979 predictions plus inference of 435 omitted events",
            },
        }
    )
    pipeline.write_json(str(Path(seed_dirs["root"]) / "seed_manifest.json"), manifest)


def write_summaries(results: list[dict], calibration_frames: list[pd.DataFrame]) -> None:
    summary_dir = Path(pipeline.SUMMARY_OUTPUT_DIR)
    summary_dir.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(results)
    result_df.to_csv(summary_dir / "resumen_seeds.csv", index=False)

    metric_columns = [
        "AUC_ROC",
        "AUC_PR",
        "F1-Score_Adv",
        "Recall_Adv",
        "Precision_Adv",
        "TN",
        "FP",
        "FN",
        "TP",
    ]
    result_df.groupby("Split")[metric_columns].agg(["mean", "std"]).to_csv(
        summary_dir / "media_std_por_split.csv"
    )

    calibration_df = pd.concat(calibration_frames, ignore_index=True)
    calibration_df.to_csv(summary_dir / "threshold_calibration_resumen_seeds.csv", index=False)
    calibration_metric_columns = [
        "Accuracy",
        "AUC_ROC",
        "AUC_PR",
        "Brier",
        "LogLoss",
        "F1_Adv",
        "Recall_Adv",
        "Specificity",
        "Precision_Adv",
    ]
    calibration_df.groupby(["Calibration", "Threshold_Strategy", "Split"])[
        calibration_metric_columns
    ].agg(["mean", "std"]).to_csv(summary_dir / "threshold_calibration_media_std.csv")


def main() -> None:
    if not SOURCE_RUN.is_dir():
        raise FileNotFoundError(f"Source run not found: {SOURCE_RUN}")

    validation_dataset = pipeline.BioCASNumpyDataset(pipeline.VALIDATION_PATH)
    test_dataset = pipeline.BioCASNumpyDataset(pipeline.TEST_PATH)
    test_counts = direct_class_counts(Path(pipeline.TEST_PATH))

    if test_counts != EXPECTED_TEST_CLASS_COUNTS:
        raise RuntimeError(
            f"Unexpected ICBHI test composition: {test_counts}; "
            f"expected {EXPECTED_TEST_CLASS_COUNTS}."
        )
    if len(test_dataset) != EXPECTED_TEST_EVENTS:
        raise RuntimeError(
            f"Incomplete corrected test loader: {len(test_dataset)} of {EXPECTED_TEST_EVENTS} events."
        )

    loaded_binary_counts = Counter(label for _, label in test_dataset.samples)
    expected_binary_counts = Counter({0: 629, 1: 785})
    if loaded_binary_counts != expected_binary_counts:
        raise RuntimeError(
            f"Unexpected binary test composition: {dict(loaded_binary_counts)}; "
            f"expected {dict(expected_binary_counts)}."
        )

    write_correction_run_manifest(len(validation_dataset), test_counts)
    crackle_samples = [
        (path, label)
        for path, label in test_dataset.samples
        if Path(path).parent.name == "Crackle"
    ]
    if len(crackle_samples) != EXPECTED_TEST_CLASS_COUNTS["Crackle"]:
        raise RuntimeError(f"Expected 435 Crackle samples, found {len(crackle_samples)}.")
    recovered_dataset = PreloadedNumpyDataset(crackle_samples)

    results: list[dict] = []
    calibration_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict] = []

    for seed in SEEDS:
        source_seed_dir = SOURCE_RUN / f"seed_{seed}"
        source_checkpoint = source_seed_dir / "checkpoints" / "best_validation_pr_auc.pth"
        source_manifest_path = source_seed_dir / "seed_manifest.json"
        if not source_checkpoint.is_file() or not source_manifest_path.is_file():
            raise FileNotFoundError(f"Missing source artifacts for seed {seed}: {source_seed_dir}")

        source_manifest = load_json(source_manifest_path)
        seed_dirs = pipeline.build_seed_output_dirs(seed)
        copied_checkpoint = Path(seed_dirs["checkpoints"]) / source_checkpoint.name
        shutil.copy2(source_checkpoint, copied_checkpoint)
        write_seed_manifest(
            seed=seed,
            seed_dirs=seed_dirs,
            source_checkpoint=source_checkpoint,
            copied_checkpoint=copied_checkpoint,
            train_events=int(source_manifest["dataset_sizes"]["train"]),
            validation_events=len(validation_dataset),
            test_events=len(test_dataset),
        )

        excel_path = Path(seed_dirs["reports"]) / "metricas.xlsx"
        validation_predictions = Path(seed_dirs["predictions"]) / "predicciones_validation.csv"
        test_predictions = Path(seed_dirs["predictions"]) / "predicciones_test.csv"
        validation_figure = Path(seed_dirs["figures"]) / "analisis_validation.png"
        test_figure = Path(seed_dirs["figures"]) / "analisis_test.png"

        print(f"\n--- Corrected evaluation | Seed {seed} ---")
        source_validation = pd.read_csv(
            source_seed_dir / "predictions" / "predicciones_validation.csv"
        )
        source_test = pd.read_csv(source_seed_dir / "predictions" / "predicciones_test.csv")
        if len(source_validation) != len(validation_dataset):
            raise RuntimeError(
                f"Seed {seed}: source validation has {len(source_validation)} events; "
                f"expected {len(validation_dataset)}."
            )
        if len(source_test) != 979:
            raise RuntimeError(f"Seed {seed}: source test has {len(source_test)} events; expected 979.")
        if Counter(source_test["y_true"].astype(int)) != Counter({0: 629, 1: 350}):
            raise RuntimeError(f"Seed {seed}: unexpected source test binary composition.")

        recovered_predictions = infer_predictions(
            checkpoint=copied_checkpoint,
            dataset=recovered_dataset,
            seed=seed,
        )
        full_test_predictions = pd.concat(
            [
                normalize_prediction_frame(source_test, seed=seed, split_name="test"),
                recovered_predictions,
            ],
            ignore_index=True,
        )
        if full_test_predictions["sample_path"].duplicated().any():
            raise RuntimeError(f"Seed {seed}: duplicate paths after completing test predictions.")
        if len(full_test_predictions) != EXPECTED_TEST_EVENTS:
            raise RuntimeError(
                f"Seed {seed}: completed test has {len(full_test_predictions)} events; "
                f"expected {EXPECTED_TEST_EVENTS}."
            )
        if Counter(full_test_predictions["y_true"].astype(int)) != expected_binary_counts:
            raise RuntimeError(f"Seed {seed}: completed test has an unexpected class composition.")

        results.append(
            report_from_predictions(
                predictions=source_validation,
                split_name="validation",
                excel_file=excel_path,
                predictions_file=validation_predictions,
                figure_file=validation_figure,
                seed=seed,
            )
        )
        results.append(
            report_from_predictions(
                predictions=full_test_predictions,
                split_name="test",
                excel_file=excel_path,
                predictions_file=test_predictions,
                figure_file=test_figure,
                seed=seed,
            )
        )
        calibration_frames.append(
            pipeline.analizar_thresholds_y_calibracion(
                validation_predictions_file=str(validation_predictions),
                test_predictions_file=str(test_predictions),
                output_dir=seed_dirs["calibration"],
                run_base_name="",
                seed=seed,
            )
        )
        coverage_rows.append(
            {
                "seed": seed,
                "source_run": project_relative(SOURCE_RUN),
                "source_test_events": int(source_manifest["dataset_sizes"]["test"]),
                "corrected_test_events": len(test_dataset),
                "normal_events": loaded_binary_counts[0],
                "adventitious_events": loaded_binary_counts[1],
                "crackle_only_events_recovered": EXPECTED_TEST_CLASS_COUNTS["Crackle"],
                "checkpoint_sha256": sha256(source_checkpoint),
            }
        )

        if pipeline.device.type == "mps" and hasattr(pipeline.torch, "mps"):
            pipeline.torch.mps.empty_cache()
        elif pipeline.device.type == "cuda":
            pipeline.torch.cuda.empty_cache()

    write_summaries(results, calibration_frames)
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_path = Path(pipeline.SUMMARY_OUTPUT_DIR) / "test_coverage_audit.csv"
    coverage_df.to_csv(coverage_path, index=False)

    print("\nCorrected evaluation completed.")
    print(f"Output run: {pipeline.MODEL_OUTPUT_DIR}")
    print(f"ICBHI test coverage: {len(test_dataset)}/{EXPECTED_TEST_EVENTS}")
    print(f"Coverage audit: {coverage_path}")


if __name__ == "__main__":
    main()
