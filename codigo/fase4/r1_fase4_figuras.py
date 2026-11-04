#!/usr/bin/env python3
"""R1 Fase 4: figuras del manuscrito revisado (R4.CS6, R1.5).

Basado en el generador de las figuras del manuscrito enviado (no incluido):
mismos datos y colores, con tipografia mayor, sin titulos internos (los da el pie) y tamano
pensado para el ancho de texto (7.0 in), de modo que las fuentes no se reducen al insertarlas.

Figuras:
  r1_fig_dataset_summary      composicion de los datasets (sustituye a la Fig. 2 del envio).
  r1_fig_primary_auroc        AUROC con IC 95 % por paciente de las tres familias principales (nueva).
  r1_fig_domain_shift         duracion (duracion real del evento, no la rellenada) + UMAP YAMNet loop
                              + UMAP wav2vec 2.0 congelado (proyeccion del envio, no depende del relleno).
  r1_fig_domain_distance      MMD y sliced Wasserstein con YAMNet loop.

Entradas (rutas relativas a la raiz del repositorio):
  analisis/r1_20261101_fase2_v2/bootstrap_por_paciente.csv
  analisis/r1_20261101_fase4/embedding_shift/{yamnet_test_embedding_projection_loop,embedding_shift_summary_loop}.csv
  analisis/r1_20261101_fase4/domain_distance/domain_distance_summary_loop.csv
  ficheros/figuras_originales/analisis_frozen/{wav2vec2_test_embedding_projection,embedding_shift_summary}.csv
  ficheros/3 Embeddings YAMNet/r1_20261029_*_looppad_mean_pool_1024/segmentado_test_metadata.csv

Uso (desde la raiz del repositorio): python codigo/fase4/r1_fase4_figuras.py
Salidas: analisis/r1_20261101_fase4/figuras/*.{pdf,png} y copia de los PDF en manuscrito/r1/Figures/
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

OUT = Path("analisis/r1_20261101_fase4/figuras")
MANUSCRIPT_FIGURES = Path("manuscrito/r1/Figures")
FIG4 = Path("analisis/r1_20261101_fase4")
ORIG = Path("ficheros/figuras_originales/analisis_frozen")
EMB = Path("ficheros/3 Embeddings YAMNet")

WIDTH = 7.0
DATASET_COLORS = {"SPRSound": "#33658A", "ICBHI": "#F4A261"}
FAMILY_COLORS = {"CNN (scratch)": "#33658A", "YAMNet trainable head": "#D1495B", "wav2vec 2.0 fine-tuned": "#6C5B7B"}
FAMILIES = {"CNN scratch": "CNN (scratch)", "YAMNet trainable head (loop pad)": "YAMNet trainable head",
            "wav2vec 2.0 fine-tuned (R1 fixed)": "wav2vec 2.0 fine-tuned"}
PROTOCOLS = {"sprsound_to_sprsound": "SPRSound → SPRSound", "icbhi_to_icbhi": "ICBHI → ICBHI",
             "icbhi_to_sprsound": "ICBHI → SPRSound", "sprsound_to_icbhi": "SPRSound → ICBHI"}


def setup() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9.5, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "legend.fontsize": 8.5, "axes.titleweight": "bold",
                         "pdf.fonttype": 42, "font.family": "DejaVu Sans"})


def panel(ax, letter: str) -> None:
    ax.set_title(letter, loc="left", fontweight="bold", fontsize=11)


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    if MANUSCRIPT_FIGURES.is_dir():
        shutil.copy2(OUT / f"{stem}.pdf", MANUSCRIPT_FIGURES / f"{stem}.pdf")


def dataset_summary() -> None:
    data = pd.DataFrame(
        [("SPRSound", "Train", 1405, 4107, 1232), ("SPRSound", "Validation", 367, 1052, 265),
         ("SPRSound", "Test", 784, 2325, 666), ("ICBHI", "Train", 535, 2292, 1890),
         ("ICBHI", "Validation", 193, 721, 581), ("ICBHI", "Test", 192, 629, 785),
         ("Fusion", "Train", 1940, 6399, 3122), ("Fusion", "Validation", 560, 1773, 846),
         ("Fusion", "Test", 976, 2954, 1451)],
        columns=["Dataset", "Split", "Recordings", "Normal", "Adventitious"])
    split_colors = {"Train": "#33658A", "Validation": "#F4A261", "Test": "#5B8E7D"}
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.4), gridspec_kw={"width_ratios": [1, 1.25]})
    sns.barplot(data=data, x="Dataset", y="Recordings", hue="Split", hue_order=list(split_colors),
                palette=split_colors, ax=axes[0])
    axes[0].set_ylabel("Recordings")
    axes[0].set_xlabel("")
    axes[0].legend(frameon=False, title=None)
    panel(axes[0], "A")

    ax = axes[1]
    y = np.arange(len(data))[::-1]
    ax.barh(y, data["Normal"], color="#33658A", label="Normal")
    ax.barh(y, data["Adventitious"], left=data["Normal"], color="#D1495B", label="Adventitious")
    for yi, row in zip(y, data.itertuples()):
        total = row.Normal + row.Adventitious
        ax.text(total + 120, yi, f"{row.Adventitious / total:.2f}", va="center", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.Dataset} {r.Split.lower()}" for r in data.itertuples()])
    ax.set_xlabel("Respiratory events")
    ax.set_xlim(0, data[["Normal", "Adventitious"]].sum(axis=1).max() * 1.14)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", visible=False)
    panel(ax, "B")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "r1_fig_dataset_summary")


def primary_auroc() -> None:
    boot = pd.read_csv("analisis/r1_20261101_fase2_v2/bootstrap_por_paciente.csv")
    df = boot[boot["model"].isin(FAMILIES)].copy()
    if len(df) != 12:
        raise ValueError("Se esperaban 12 filas")
    df["family"] = df["model"].map(FAMILIES)
    fig, ax = plt.subplots(figsize=(WIDTH, 3.3))
    offsets = {f: o for f, o in zip(FAMILY_COLORS, (-0.22, 0.0, 0.22))}
    order = list(PROTOCOLS)
    for family, color in FAMILY_COLORS.items():
        sub = df[df["family"] == family].set_index("protocol").reindex(order)
        x = np.arange(len(order)) + offsets[family]
        ax.errorbar(x, sub["auroc"], yerr=[sub["auroc"] - sub["auroc_patient_lo"], sub["auroc_patient_hi"] - sub["auroc"]],
                    fmt="o", color=color, ecolor=color, elinewidth=1.4, capsize=3, markersize=6, label=family)
    ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1)
    ax.text(len(order) - 0.52, 0.505, "chance", fontsize=8.5, color="#555555", va="bottom", ha="right")
    ax.axvline(1.5, color="#BBBBBB", linewidth=0.8)
    ax.text(0.5, 0.955, "Within dataset", ha="center", fontsize=9, transform=ax.get_xaxis_transform())
    ax.text(2.5, 0.955, "Cross dataset", ha="center", fontsize=9, transform=ax.get_xaxis_transform())
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([PROTOCOLS[p] for p in order])
    ax.set_ylabel("Test AUROC (95% CI)")
    ax.set_ylim(0.15, 1.0)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    save(fig, "r1_fig_primary_auroc")


def domain_shift() -> None:
    durations = []
    for key, name in (("sprsound", "SPRSound"), ("icbhi", "ICBHI")):
        meta = pd.read_csv(EMB / f"r1_20261029_{key}_seed42_yamnet_looppad_mean_pool_1024/segmentado_test_metadata.csv")
        durations.append(pd.DataFrame({"Dataset": name, "Event label": meta["label_binary"].map({0: "Normal", 1: "Adventitious"}),
                                       "Duration (s)": meta["original_duration_seconds"]}))
    duration = pd.concat(durations, ignore_index=True)
    yamnet = pd.read_csv(FIG4 / "embedding_shift/yamnet_test_embedding_projection_loop.csv")
    yamnet_scores = pd.read_csv(FIG4 / "embedding_shift/embedding_shift_summary_loop.csv").iloc[0]
    wav2vec = pd.read_csv(ORIG / "wav2vec2_test_embedding_projection.csv")
    summary = pd.read_csv(ORIG / "embedding_shift_summary.csv")
    wav2vec_scores = summary[(summary["model_family"] == "wav2vec2") & (summary["projection_method"] == "umap")].iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.9), gridspec_kw={"width_ratios": [1.05, 1, 1]})
    sns.violinplot(data=duration, x="Event label", y="Duration (s)", hue="Dataset", order=["Normal", "Adventitious"],
                   hue_order=list(DATASET_COLORS), palette=DATASET_COLORS, cut=0, inner="quartile",
                   density_norm="width", linewidth=0.8, ax=axes[0])
    axes[0].set_xlabel("")
    axes[0].set_ylim(0, 8)
    panel(axes[0], "A")
    for ax, frame, scores, letter, name in [(axes[1], yamnet, yamnet_scores, "B", "YAMNet"),
                                            (axes[2], wav2vec, wav2vec_scores, "C", "wav2vec 2.0")]:
        for key, label in (("sprsound", "SPRSound"), ("icbhi", "ICBHI")):
            sub = frame[frame["dataset_key"] == key]
            ax.scatter(sub["coord_1"], sub["coord_2"], s=3, alpha=0.4, color=DATASET_COLORS[label], linewidth=0,
                       label=label, rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"UMAP 1 ({name})\nsil. dataset {scores['dataset_silhouette']:.2f}, "
                      f"class {scores['class_silhouette']:.2f}", fontsize=8.5)
        ax.set_ylabel("UMAP 2")
        panel(ax, letter)
    axes[0].get_legend().remove()
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=8, color=c) for c in DATASET_COLORS.values()]
    fig.legend(handles, list(DATASET_COLORS), loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save(fig, "r1_fig_domain_shift")


def domain_distance() -> None:
    distance = pd.read_csv(FIG4 / "domain_distance/domain_distance_summary_loop.csv")
    display = {"cnn_penultimate": "CNN (scratch), penultimate layer", "yamnet_frozen_embedding": "Frozen YAMNet embedding",
               "yamnet_trainable_head": "YAMNet trainable head, hidden layer"}
    colors = ["#33658A", "#5B8E7D", "#D1495B"]
    conditions = {"overall": "All events", "healthy": "Normal", "adventitious": "Adventitious"}
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0))
    width = 0.26
    for ax, metric, ylabel, letter in [(axes[0], "rbf_mmd", "RBF MMD", "A"),
                                       (axes[1], "sliced_wasserstein", "Sliced Wasserstein distance", "B")]:
        x = np.arange(len(conditions))
        for i, rep in enumerate(display):
            sub = distance[distance["representation"] == rep].set_index("condition").reindex(list(conditions))
            ax.bar(x + (i - 1) * width, sub[f"{metric}_mean"], width=width, yerr=sub[f"{metric}_std"], capsize=2,
                   color=colors[i], edgecolor="white", linewidth=0.6, label=display[rep])
        ax.set_xticks(x)
        ax.set_xticklabels(list(conditions.values()))
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
        ax.spines[["top", "right"]].set_visible(False)
        panel(ax, letter)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "r1_fig_domain_distance")


def main() -> None:
    setup()
    dataset_summary()
    primary_auroc()
    domain_shift()
    domain_distance()
    print("Figuras en", OUT)


if __name__ == "__main__":
    main()
