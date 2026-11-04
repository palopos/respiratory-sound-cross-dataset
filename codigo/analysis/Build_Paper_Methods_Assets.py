from __future__ import annotations

import json
import os
from pathlib import Path


def _find_project_root(start_dir: str) -> Path:
    current_dir = Path(start_dir).resolve()
    while True:
        if (current_dir / "environment.yml").exists() and (current_dir / "ficheros").is_dir():
            return current_dir
        if current_dir.parent == current_dir:
            raise FileNotFoundError("Could not locate the Fix_Paper1 project root.")
        current_dir = current_dir.parent


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _find_project_root(str(SCRIPT_DIR))
os.chdir(PROJECT_ROOT)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


OUTPUT_DIR = Path("./modelos/comparativas/paper_final")
FIGURES_DIR = OUTPUT_DIR / "figures"

SPLIT_MANIFESTS = {
    "SPRSound": Path("./ficheros/splits/0 SPRSound/splits_patient_independent_seed42.json"),
    "ICBHI": Path("./ficheros/splits/0 ICBHI/splits_patient_independent_seed42.json"),
    "Fusion": Path("./ficheros/splits/0 Fusion SPRSound ICBHI/splits_patient_independent_seed42.json"),
}

FIGURE_FILES = {
    "study_design": "figure_methods_study_design_overview",
    "dataset_summary": "figure_methods_dataset_summary",
    "model_taxonomy": "figure_methods_model_taxonomy",
}

COLORS = {
    "navy": "#1f3b73",
    "blue": "#3a7bd5",
    "sky": "#74add1",
    "teal": "#2a9d8f",
    "green": "#5aa469",
    "orange": "#f4a261",
    "red": "#e76f51",
    "gold": "#e9c46a",
    "gray": "#6c757d",
    "light_gray": "#edf2f7",
    "light_blue": "#e8f1fb",
    "light_green": "#e7f4ef",
    "light_orange": "#fff1e6",
    "light_red": "#fdecea",
}


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_manifests() -> dict[str, dict]:
    return {
        dataset_name: json.loads(path.read_text(encoding="utf-8"))
        for dataset_name, path in SPLIT_MANIFESTS.items()
    }


def build_dataset_summary(manifests: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for dataset_name, payload in manifests.items():
        for split_name, summary in payload["summary"].items():
            rows.append(
                {
                    "dataset": dataset_name,
                    "split": split_name,
                    "recordings": int(summary["recordings"]),
                    "patients_with_id": int(summary["patients_with_id"]),
                    "events_total": int(summary["events_total"]),
                    "normal_events": int(summary["normal_events"]),
                    "adventitious_events": int(summary["adventitious_events"]),
                    "adventitious_ratio": float(summary["adventitious_events"]) / float(summary["events_total"]),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "paper_methods_dataset_summary.csv", index=False)
    return df


def build_model_summary() -> pd.DataFrame:
    rows = [
        {
            "family": "CNN scratch",
            "input_representation": "64x64 log-Mel spectrogram",
            "input_source": "Segmented respiratory events",
            "sample_rate_hz": 8000,
            "feature_detail": "64 mel bins; dynamic resize; Butterworth band-pass 50-3999 Hz",
            "training_regime": "End-to-end training from scratch",
            "model_detail": "BioCAS-inspired grouped CNN with SE attention and 2-layer MLP classifier",
            "selection_rule": "Best validation PR-AUC checkpoint",
            "default_hyperparameters": "batch=32; epochs=30; lr=1e-4; class weights=[1.0, 3.5]",
            "study_role": "Main SPRSound baseline",
        },
        {
            "family": "YAMNet frozen + LR",
            "input_representation": "1024-d mean-pooled embedding",
            "input_source": "Segmented respiratory events",
            "sample_rate_hz": 16000,
            "feature_detail": "TF-Hub YAMNet embeddings; min input 0.975 s with zero-padding",
            "training_regime": "Frozen embedding baseline",
            "model_detail": "StandardScaler + LogisticRegression",
            "selection_rule": "Deterministic baseline; no checkpointing",
            "default_hyperparameters": "C=1.0; solver=liblinear; max_iter=2000",
            "study_role": "Pretrained baseline",
        },
        {
            "family": "YAMNet frozen + trainable head",
            "input_representation": "1024-d mean-pooled embedding",
            "input_source": "Precomputed YAMNet embeddings",
            "sample_rate_hz": 16000,
            "feature_detail": "Same frozen YAMNet backbone as above",
            "training_regime": "Trainable MLP head only",
            "model_detail": "Normalization + Dense(256) + Dense(64) + sigmoid output",
            "selection_rule": "Best validation ROC-AUC checkpoint",
            "default_hyperparameters": "batch=128; epochs=20; patience=4; lr=1e-3; dropout=0.30",
            "study_role": "Main ICBHI and ICBHI->SPRSound model",
        },
        {
            "family": "wav2vec2 frozen + LR",
            "input_representation": "768-d mean-pooled embedding",
            "input_source": "Segmented respiratory events",
            "sample_rate_hz": 16000,
            "feature_detail": "facebook/wav2vec2-base embeddings",
            "training_regime": "Frozen embedding baseline",
            "model_detail": "StandardScaler + LogisticRegression",
            "selection_rule": "Deterministic baseline; no checkpointing",
            "default_hyperparameters": "C=1.0; solver=liblinear; max_iter=2000",
            "study_role": "Explored but deprioritized",
        },
        {
            "family": "wav2vec2 fine-tuning",
            "input_representation": "Raw waveform",
            "input_source": "Segmented respiratory events",
            "sample_rate_hz": 16000,
            "feature_detail": "facebook/wav2vec2-base sequence classifier",
            "training_regime": "Full model fine-tuning screening",
            "model_detail": "Wav2Vec2ForSequenceClassification",
            "selection_rule": "Best validation ROC-AUC then PR-AUC",
            "default_hyperparameters": "train_batch=4; eval_batch=8; epochs=8; patience=3; lr=1e-5",
            "study_role": "Explored but deprioritized",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "paper_methods_model_summary.csv", index=False)
    return df


def style_axis_clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_box(ax, xy, width, height, text, facecolor, edgecolor, fontsize=10.5, lw=1.8):
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=lw,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2.0,
        xy[1] + height / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#1b1f23",
        wrap=True,
    )


def add_arrow(ax, start, end, color="#4a5568", lw=1.7):
    arrow = patches.FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=lw,
        color=color,
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arrow)


def save_figure(fig: plt.Figure, basename: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{basename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{basename}.pdf", bbox_inches="tight")
    plt.close(fig)


def build_study_design_figure() -> None:
    fig, ax = plt.subplots(figsize=(16.5, 8.8))
    style_axis_clean(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.03, 0.95, "Study design and leakage-controlled evaluation protocol", fontsize=18, fontweight="bold", color=COLORS["navy"])
    ax.text(
        0.03,
        0.91,
        "The entire study was structured around patient-independent splitting, validation-only model selection, and strictly isolated test evaluation.",
        fontsize=11.5,
        color="#38424d",
    )

    # Column anchors
    col_x = [0.03, 0.225, 0.42, 0.615, 0.81]
    width = 0.155
    height = 0.17

    add_box(
        ax,
        (col_x[0], 0.63),
        width,
        height,
        "Source datasets\n\nSPRSound\nICBHI\nFusion\n(concatenated clean splits)",
        COLORS["light_blue"],
        COLORS["blue"],
    )
    add_box(
        ax,
        (col_x[1], 0.63),
        width,
        height,
        "Split construction\n\nPatient-independent\ntrain/validation/test\nExclude Poor Quality\nExclude missing patient IDs\nStore split manifests",
        COLORS["light_green"],
        COLORS["green"],
        fontsize=9.6,
    )
    add_box(
        ax,
        (col_x[2], 0.63),
        width,
        height,
        "Event preprocessing\n\nSegment events from annotations\nSave split-specific WAVs\nDerive 8-kHz spectrograms\nor 16-kHz embeddings",
        COLORS["light_orange"],
        COLORS["orange"],
        fontsize=9.6,
    )
    add_box(
        ax,
        (col_x[3], 0.63),
        width,
        height,
        "Model families\n\nCNN scratch\nYAMNet frozen + LR\nYAMNet trainable head\nwav2vec2 frozen / fine-tuned",
        COLORS["light_red"],
        COLORS["red"],
        fontsize=9.6,
    )
    add_box(
        ax,
        (col_x[4], 0.63),
        width,
        height,
        "Selection and reporting\n\nCheckpoint from validation only\nThreshold from validation only\nTest used once for final reporting",
        COLORS["light_gray"],
        COLORS["gray"],
        fontsize=9.6,
    )

    for idx in range(4):
        add_arrow(ax, (col_x[idx] + width, 0.71), (col_x[idx + 1], 0.71))

    add_box(
        ax,
        (0.08, 0.28),
        0.24,
        0.18,
        "Primary comparison grid\n\nIntra-dataset:\nSPRSound->SPRSound\nICBHI->ICBHI\n\nCross-dataset:\nICBHI->SPRSound\nSPRSound->ICBHI",
        "#f7fafc",
        COLORS["navy"],
        fontsize=9.9,
    )
    add_box(
        ax,
        (0.39, 0.28),
        0.24,
        0.18,
        "Follow-up analyses\n\nFrozen vs trainable heads\nThreshold / calibration analysis\nBinary mapping validation\nTargeted augmentation\nCORAL screening",
        "#f7fafc",
        COLORS["navy"],
        fontsize=9.9,
    )
    add_box(
        ax,
        (0.70, 0.28),
        0.22,
        0.18,
        "Final paper shortlist\n\nCNN scratch for\nSPRSound intra-dataset\n\nYAMNet trainable head for\nICBHI intra-dataset and\nICBHI->SPRSound\n\nMixup retained as a\nsecondary refinement",
        "#f7fafc",
        COLORS["navy"],
        fontsize=9.8,
    )

    add_arrow(ax, (0.15, 0.63), (0.20, 0.46), color=COLORS["navy"])
    add_arrow(ax, (0.50, 0.63), (0.51, 0.46), color=COLORS["navy"])
    add_arrow(ax, (0.79, 0.63), (0.81, 0.46), color=COLORS["navy"])

    ax.text(
        0.03,
        0.11,
        "Key guardrails applied throughout: no patient overlap across splits, validation-derived checkpoint/threshold selection, "
        "explicit storage of predictions and probabilities, and statistical validation on the final shortlist.",
        fontsize=11.2,
        color="#38424d",
    )

    save_figure(fig, FIGURE_FILES["study_design"])


def build_dataset_summary_figure(dataset_df: pd.DataFrame) -> None:
    order = ["SPRSound", "ICBHI", "Fusion"]
    split_order = ["train", "validation", "test"]
    split_colors = {
        "train": COLORS["blue"],
        "validation": COLORS["gold"],
        "test": COLORS["red"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2))

    # Panel A: recordings
    ax = axes[0]
    x = np.arange(len(order))
    width = 0.22
    for idx, split_name in enumerate(split_order):
        subset = dataset_df[dataset_df["split"] == split_name].set_index("dataset").loc[order]
        ax.bar(
            x + (idx - 1) * width,
            subset["recordings"],
            width=width,
            color=split_colors[split_name],
            label=split_name.capitalize(),
            edgecolor="white",
        )
        for x_i, value in zip(x + (idx - 1) * width, subset["recordings"]):
            ax.text(x_i, value + 22, f"{int(value)}", ha="center", va="bottom", fontsize=9)

    ax.set_title("A. Recordings per split", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Number of recordings")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")

    # Panel B: events with class breakdown
    ax = axes[1]
    positions = []
    labels = []
    normal_counts = []
    adventitious_counts = []
    for dataset_name in order:
        for split_name in split_order:
            row = dataset_df[(dataset_df["dataset"] == dataset_name) & (dataset_df["split"] == split_name)].iloc[0]
            positions.append(len(positions))
            labels.append(f"{dataset_name}\n{split_name}")
            normal_counts.append(row["normal_events"])
            adventitious_counts.append(row["adventitious_events"])

    positions = np.asarray(positions)
    ax.bar(positions, normal_counts, color=COLORS["teal"], label="Normal events", edgecolor="white")
    ax.bar(positions, adventitious_counts, bottom=normal_counts, color=COLORS["orange"], label="Adventitious events", edgecolor="white")

    totals = np.asarray(normal_counts) + np.asarray(adventitious_counts)
    for pos, total in zip(positions, totals):
        ax.text(pos, total + 55, f"{int(total)}", ha="center", va="bottom", fontsize=8.5, rotation=90)

    ax.set_title("B. Event counts and class balance", fontsize=13, fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0, fontsize=8.7)
    ax.set_ylabel("Number of events")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")

    fig.suptitle("Dataset composition after protocol cleanup", fontsize=17, fontweight="bold", color=COLORS["navy"])
    save_figure(fig, FIGURE_FILES["dataset_summary"])


def build_model_taxonomy_figure() -> None:
    fig, ax = plt.subplots(figsize=(15.5, 8.4))
    style_axis_clean(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.03, 0.95, "Model taxonomy evaluated in the study", fontsize=18, fontweight="bold", color=COLORS["navy"])
    ax.text(
        0.03,
        0.91,
        "The experimental grid combined a dataset-specific CNN, frozen pretrained baselines, and trainable variants screened under the same split protocol.",
        fontsize=11.5,
        color="#38424d",
    )

    # Input lane
    add_box(
        ax,
        (0.03, 0.62),
        0.20,
        0.20,
        "Segmented event WAVs\n\nPatient-independent splits\nreused across all\nmodel families",
        COLORS["light_gray"],
        COLORS["gray"],
        fontsize=10.5,
    )
    add_box(ax, (0.27, 0.70), 0.18, 0.12, "8-kHz log-Mel branch\n64 x 64 spectrogram", COLORS["light_blue"], COLORS["blue"])
    add_box(ax, (0.27, 0.50), 0.18, 0.12, "16-kHz embedding branch\nYAMNet or wav2vec2", COLORS["light_green"], COLORS["green"])
    add_arrow(ax, (0.23, 0.72), (0.27, 0.76))
    add_arrow(ax, (0.23, 0.72), (0.27, 0.56))

    add_box(ax, (0.52, 0.72), 0.18, 0.10, "CNN scratch\nMain SPRSound model", COLORS["light_red"], COLORS["red"])
    add_arrow(ax, (0.45, 0.76), (0.52, 0.77), color=COLORS["red"])

    add_box(ax, (0.52, 0.56), 0.18, 0.10, "YAMNet frozen + LR\nDeterministic baseline", COLORS["light_green"], COLORS["green"])
    add_box(ax, (0.52, 0.42), 0.18, 0.10, "YAMNet frozen + trainable head\nMain ICBHI / transfer model", "#dff3e3", "#2f855a")
    add_box(ax, (0.52, 0.28), 0.18, 0.10, "wav2vec2 frozen + LR\nBaseline only", "#edf2ff", "#667eea")
    add_box(ax, (0.52, 0.14), 0.18, 0.10, "wav2vec2 fine-tuning\nScreening only", "#f3f0ff", "#805ad5")

    add_arrow(ax, (0.45, 0.56), (0.52, 0.61), color=COLORS["green"])
    add_arrow(ax, (0.45, 0.56), (0.52, 0.47), color="#2f855a")
    add_arrow(ax, (0.45, 0.56), (0.52, 0.33), color="#667eea")
    add_arrow(ax, (0.45, 0.56), (0.52, 0.19), color="#805ad5")

    add_box(ax, (0.77, 0.60), 0.18, 0.22, "Shortlisted outcomes\n\nSPRSound -> SPRSound:\nCNN scratch\n\nICBHI -> ICBHI:\nYAMNet trainable head\n\nICBHI -> SPRSound:\nYAMNet trainable head", "#fffbea", "#b7791f", fontsize=10.6)
    add_box(ax, (0.77, 0.28), 0.18, 0.22, "Secondary / negative findings\n\nMixup: small secondary gain\nCORAL: degraded transfer\nwav2vec2: deprioritized", "#f7fafc", COLORS["gray"], fontsize=10.6)

    add_arrow(ax, (0.70, 0.77), (0.77, 0.71), color="#4a5568")
    add_arrow(ax, (0.70, 0.61), (0.77, 0.39), color="#4a5568")
    add_arrow(ax, (0.70, 0.47), (0.77, 0.39), color="#4a5568")
    add_arrow(ax, (0.70, 0.33), (0.77, 0.39), color="#4a5568")
    add_arrow(ax, (0.70, 0.19), (0.77, 0.39), color="#4a5568")

    ax.text(
        0.03,
        0.05,
        "Shortlist promotion required all of the following: validation-only model selection, isolated test evaluation, statistical validation, and targeted ablations that did not overturn the main conclusion.",
        fontsize=11.0,
        color="#38424d",
    )

    save_figure(fig, FIGURE_FILES["model_taxonomy"])


def write_manifest(dataset_df: pd.DataFrame, model_df: pd.DataFrame) -> None:
    def dataframe_to_bullets(df: pd.DataFrame) -> list[str]:
        lines = []
        for _, row in df.iterrows():
            pairs = [f"{column}={row[column]}" for column in df.columns]
            lines.append("- " + "; ".join(pairs))
        return lines

    lines = [
        "# Paper Methods Assets",
        "",
        "Generated figures:",
        "",
        f"- `{FIGURE_FILES['study_design']}.pdf/.png`: organigram of the full experimental pipeline.",
        f"- `{FIGURE_FILES['dataset_summary']}.pdf/.png`: dataset composition after protocol cleanup.",
        f"- `{FIGURE_FILES['model_taxonomy']}.pdf/.png`: taxonomy of the evaluated model families.",
        "",
        "Generated tables (CSV):",
        "",
        "- `paper_methods_dataset_summary.csv`: recordings, patients, events, and class balance per dataset split.",
        "- `paper_methods_model_summary.csv`: consolidated summary of model families and default training settings.",
        "",
        "Dataset summary preview:",
        "",
        *dataframe_to_bullets(dataset_df),
        "",
        "Model summary preview:",
        "",
        *dataframe_to_bullets(model_df),
    ]
    (OUTPUT_DIR / "paper_methods_figures_manifest.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_output_dirs()
    manifests = load_manifests()
    dataset_df = build_dataset_summary(manifests)
    model_df = build_model_summary()
    build_study_design_figure()
    build_dataset_summary_figure(dataset_df)
    build_model_taxonomy_figure()
    write_manifest(dataset_df, model_df)
    print(f">>> Paper methods assets saved to: {OUTPUT_DIR}")
    print(f">>> Figures directory: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
