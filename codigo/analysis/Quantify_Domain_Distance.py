from __future__ import annotations

import json
import math
import os
from pathlib import Path


def _find_project_root(start_dir: str) -> Path:
    current_dir = Path(start_dir).resolve()
    while True:
        if (current_dir / "environment.yml").exists() and (current_dir / "ficheros").is_dir():
            return current_dir
        if current_dir.parent == current_dir:
            raise FileNotFoundError("No se pudo localizar la raiz de Fix_Paper1.")
        current_dir = current_dir.parent


_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _find_project_root(str(_SCRIPT_DIR))
os.chdir(_PROJECT_ROOT)
_CACHE_DIR = _PROJECT_ROOT / ".cache"
_MPLCONFIGDIR = _PROJECT_ROOT / ".matplotlib"
_NUMBA_CACHE_DIR = _CACHE_DIR / "numba"
_FONTCONFIG_PATH = _CACHE_DIR / "fontconfig"
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("NUMBA_CACHE_DIR", str(_NUMBA_CACHE_DIR))
os.environ.setdefault("FONTCONFIG_PATH", str(_FONTCONFIG_PATH))
_CACHE_DIR.mkdir(exist_ok=True)
_MPLCONFIGDIR.mkdir(exist_ok=True)
_NUMBA_CACHE_DIR.mkdir(exist_ok=True)
_FONTCONFIG_PATH.mkdir(exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler


ANALYSIS_SEED = 12345
MAX_SAMPLES_PER_GROUP = 1500
MAX_MEDIAN_HEURISTIC_SAMPLES = 1000
PCA_COMPONENTS = 32
SLICED_WASSERSTEIN_PROJECTIONS = 128

OUTPUT_DIR = Path("./modelos/comparativas/domain_distance")
PAPER_FINAL_DIR = Path("./modelos/comparativas/paper_final")
PAPER_FIGURES_DIR = PAPER_FINAL_DIR / "figures"

YAMNET_FROZEN_DIRS = {
    "source_train": Path("./ficheros/3 Embeddings YAMNet/0 ICBHI_seed42_yamnet_tfhub_mean_pool_1024"),
    "target_test": Path("./ficheros/3 Embeddings YAMNet/0 SPRSound_seed42_yamnet_tfhub_mean_pool_1024"),
}
YAMNET_TRAINABLE_RUN_DIR = Path(
    "./modelos/04-yamnet-trainable-head-confirm/icbhi-to-sprsound/"
    "yamnet/yamnet_google-yamnet-tfhub_frozen-backbone-trainable-head_trainable-mlp_"
    "classification-head-only/split_seed_42/run_20260527_072855"
)
CNN_RUN_DIR = Path(
    "./modelos/00-cnn-scratch-baseline/icbhi-to-sprsound/"
    "cnn/cnn_biocas-resnext-attention_scratch_trainable-mlp/split_seed_42/run_20260521_132020"
)
CNN_SOURCE_TRAIN_DIR = Path("./ficheros/2 Espectrogramas/0 ICBHI_seed42_resize_dinamico_MEL64_SIZE64/npy/segmentado_train")
CNN_TARGET_TEST_DIR = Path("./ficheros/2 Espectrogramas/0 SPRSound_seed42_resize_dinamico_MEL64_SIZE64/npy/segmentado_test")


def ensure_output_dirs() -> None:
    for path in (
        OUTPUT_DIR,
        OUTPUT_DIR / "tables",
        OUTPUT_DIR / "figures",
        PAPER_FIGURES_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def progress(iterable, **kwargs):
    try:
        from tqdm import tqdm

        return tqdm(iterable, **kwargs)
    except Exception:
        return iterable


def normalize_event_label(label: str) -> str:
    return str(label).strip().lower().replace(" ", "_")


def binary_label_from_name(label_name: str) -> int:
    return 0 if normalize_event_label(label_name) == "normal" else 1


def load_embedding_block(base_dir: Path, split_name: str) -> tuple[np.ndarray, pd.DataFrame]:
    payload = np.load(base_dir / f"{split_name}_embeddings.npz", allow_pickle=False)
    metadata = pd.read_csv(base_dir / f"{split_name}_metadata.csv")
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    if len(metadata) != len(embeddings):
        raise ValueError(f"Metadata and embeddings length mismatch for {base_dir} / {split_name}")
    return embeddings, metadata


def load_yamnet_frozen_representation() -> dict[str, dict]:
    source_x, source_meta = load_embedding_block(YAMNET_FROZEN_DIRS["source_train"], "segmentado_train")
    target_x, target_meta = load_embedding_block(YAMNET_FROZEN_DIRS["target_test"], "segmentado_test")
    return {
        "source": {"x": source_x, "meta": source_meta},
        "target": {"x": target_x, "meta": target_meta},
    }


def build_yamnet_feature_extractor(model_path: Path) -> tf.keras.Model:
    model = tf.keras.models.load_model(model_path, compile=False)
    feature_layer = model.get_layer("dense_2")
    return tf.keras.Model(inputs=model.input, outputs=feature_layer.output, name="yamnet_head_features")


def load_yamnet_trainable_head_representations() -> list[dict]:
    source_x, source_meta = load_embedding_block(YAMNET_FROZEN_DIRS["source_train"], "segmentado_train")
    target_x, target_meta = load_embedding_block(YAMNET_FROZEN_DIRS["target_test"], "segmentado_test")
    rows = []
    for seed_dir in sorted(YAMNET_TRAINABLE_RUN_DIR.glob("seed_*")):
        seed = int(seed_dir.name.replace("seed_", ""))
        model_path = seed_dir / "checkpoints" / "best_model.keras"
        feature_model = build_yamnet_feature_extractor(model_path)
        source_features = feature_model.predict(source_x, batch_size=512, verbose=0).astype(np.float32)
        target_features = feature_model.predict(target_x, batch_size=512, verbose=0).astype(np.float32)
        rows.append(
            {
                "representation": "yamnet_trainable_head",
                "seed": seed,
                "source": {"x": source_features, "meta": source_meta.copy()},
                "target": {"x": target_features, "meta": target_meta.copy()},
            }
        )
    return rows


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BioCASResNextBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
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


class BioCASAdvancedCNN(nn.Module):
    def __init__(self, dropout: float = 0.2, groups: int = 4):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = BioCASResNextBlock(64, 64, stride=1, groups=groups)
        self.layer2 = BioCASResNextBlock(64, 128, stride=2, groups=groups)
        self.layer3 = BioCASResNextBlock(128, 256, stride=2, groups=groups)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward_features(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = self.classifier[0](x)
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        x = self.classifier[3](x)
        return x

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        return self.classifier(x)


def iter_npy_samples(base_dir: Path):
    for class_dir in sorted(base_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label_name = class_dir.name
        label_binary = binary_label_from_name(label_name)
        for npy_path in sorted(class_dir.glob("*.npy")):
            yield {
                "sample_path": str(npy_path),
                "segment_file": npy_path.name.replace(".npy", ".wav"),
                "label_name": label_name,
                "label_binary": label_binary,
            }


def load_cnn_input_block(base_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    records = list(iter_npy_samples(base_dir))
    arrays = []
    for record in progress(records, desc=f"Loading {base_dir.name}", unit="npy", leave=False):
        arrays.append(np.load(record["sample_path"]).astype(np.float32))
    x = np.stack(arrays, axis=0)[:, None, :, :]
    metadata = pd.DataFrame(records)
    return x, metadata


def extract_cnn_features(model_path: Path, x: np.ndarray, batch_size: int = 128) -> np.ndarray:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = BioCASAdvancedCNN(dropout=0.2, groups=4).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    features = []
    with torch.no_grad():
        for start in progress(range(0, len(x), batch_size), desc=f"Extracting {model_path.parent.parent.name}", unit="batch", leave=False):
            end = start + batch_size
            batch = torch.from_numpy(x[start:end]).to(device)
            batch_features = model.forward_features(batch).cpu().numpy().astype(np.float32)
            features.append(batch_features)
    return np.concatenate(features, axis=0)


def load_cnn_representations() -> list[dict]:
    source_x, source_meta = load_cnn_input_block(CNN_SOURCE_TRAIN_DIR)
    target_x, target_meta = load_cnn_input_block(CNN_TARGET_TEST_DIR)
    rows = []
    for seed_dir in sorted(CNN_RUN_DIR.glob("seed_*")):
        seed = int(seed_dir.name.replace("seed_", ""))
        model_path = seed_dir / "checkpoints" / "best_validation_pr_auc.pth"
        source_features = extract_cnn_features(model_path, source_x)
        target_features = extract_cnn_features(model_path, target_x)
        rows.append(
            {
                "representation": "cnn_penultimate",
                "seed": seed,
                "source": {"x": source_features, "meta": source_meta.copy()},
                "target": {"x": target_features, "meta": target_meta.copy()},
            }
        )
    return rows


def subsample_for_condition(source_x, source_y, target_x, target_y, condition: str, rng: np.random.Generator):
    if condition == "overall":
        source_mask = np.ones(len(source_y), dtype=bool)
        target_mask = np.ones(len(target_y), dtype=bool)
    elif condition == "healthy":
        source_mask = source_y == 0
        target_mask = target_y == 0
    elif condition == "adventitious":
        source_mask = source_y == 1
        target_mask = target_y == 1
    else:
        raise ValueError(f"Unsupported condition: {condition}")

    src = np.asarray(source_x[source_mask], dtype=np.float32)
    tgt = np.asarray(target_x[target_mask], dtype=np.float32)
    src_y = np.asarray(source_y[source_mask], dtype=np.int64)
    tgt_y = np.asarray(target_y[target_mask], dtype=np.int64)

    n = min(len(src), len(tgt), MAX_SAMPLES_PER_GROUP)
    if n < 2:
        raise ValueError(f"Not enough samples for condition={condition}: source={len(src)} target={len(tgt)}")

    if len(src) > n:
        src_idx = rng.choice(len(src), size=n, replace=False)
        src, src_y = src[src_idx], src_y[src_idx]
    if len(tgt) > n:
        tgt_idx = rng.choice(len(tgt), size=n, replace=False)
        tgt, tgt_y = tgt[tgt_idx], tgt_y[tgt_idx]
    return src, src_y, tgt, tgt_y


def project_features(source_x: np.ndarray, target_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pooled = np.concatenate([source_x, target_x], axis=0)
    scaler = StandardScaler()
    pooled = scaler.fit_transform(pooled)
    n_components = min(PCA_COMPONENTS, pooled.shape[1], pooled.shape[0] - 1)
    if n_components >= 2 and n_components < pooled.shape[1]:
        pca = PCA(n_components=n_components, random_state=ANALYSIS_SEED)
        pooled = pca.fit_transform(pooled)
    source_proj = pooled[: len(source_x)]
    target_proj = pooled[len(source_x) :]
    return source_proj.astype(np.float32), target_proj.astype(np.float32)


def estimate_gamma_median_heuristic(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> float:
    z = np.concatenate([x, y], axis=0)
    if len(z) > MAX_MEDIAN_HEURISTIC_SAMPLES:
        idx = rng.choice(len(z), size=MAX_MEDIAN_HEURISTIC_SAMPLES, replace=False)
        z = z[idx]
    sq_dists = pdist(z, metric="sqeuclidean")
    median_sq = float(np.median(sq_dists)) if len(sq_dists) else 1.0
    median_sq = max(median_sq, 1e-6)
    return 1.0 / (2.0 * median_sq)


def rbf_mmd(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> float:
    gamma = estimate_gamma_median_heuristic(x, y, rng)
    k_xx = rbf_kernel(x, x, gamma=gamma)
    k_yy = rbf_kernel(y, y, gamma=gamma)
    k_xy = rbf_kernel(x, y, gamma=gamma)
    n = len(x)
    m = len(y)
    xx = (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
    yy = (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
    xy = k_xy.mean()
    mmd2 = float(xx + yy - 2.0 * xy)
    return math.sqrt(max(mmd2, 0.0))


def sliced_wasserstein(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_projections: int = SLICED_WASSERSTEIN_PROJECTIONS) -> float:
    dim = x.shape[1]
    directions = rng.normal(size=(n_projections, dim)).astype(np.float32)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.clip(norms, 1e-12, None)
    x_proj = x @ directions.T
    y_proj = y @ directions.T
    distances = [
        wasserstein_distance(x_proj[:, i], y_proj[:, i])
        for i in range(n_projections)
    ]
    return float(np.mean(distances))


def compute_condition_metrics(source_x, source_y, target_x, target_y, condition: str, seed_for_sampling: int):
    rng = np.random.default_rng(seed_for_sampling)
    src, src_y, tgt, tgt_y = subsample_for_condition(source_x, source_y, target_x, target_y, condition, rng)
    src_proj, tgt_proj = project_features(src, tgt)
    pooled = np.concatenate([src_proj, tgt_proj], axis=0)
    domain_codes = np.concatenate([np.zeros(len(src_proj), dtype=int), np.ones(len(tgt_proj), dtype=int)])
    class_codes = np.concatenate([src_y, tgt_y], axis=0)

    metrics = {
        "n_source": int(len(src_proj)),
        "n_target": int(len(tgt_proj)),
        "rbf_mmd": rbf_mmd(src_proj, tgt_proj, rng),
        "sliced_wasserstein": sliced_wasserstein(src_proj, tgt_proj, rng),
        "domain_centroid_distance": float(np.linalg.norm(src_proj.mean(axis=0) - tgt_proj.mean(axis=0))),
    }
    if condition == "overall" and len(np.unique(domain_codes)) == 2 and len(np.unique(class_codes)) == 2:
        metrics["domain_silhouette"] = float(silhouette_score(pooled, domain_codes))
        metrics["class_silhouette"] = float(silhouette_score(pooled, class_codes))
    else:
        metrics["domain_silhouette"] = np.nan
        metrics["class_silhouette"] = np.nan
    return metrics


def collect_all_representations() -> list[dict]:
    representations = [
        {
            "representation": "yamnet_frozen_embedding",
            "seed": 42,
            **load_yamnet_frozen_representation(),
        }
    ]
    representations.extend(load_yamnet_trainable_head_representations())
    representations.extend(load_cnn_representations())
    return representations


def quantify_domain_distance() -> pd.DataFrame:
    rows = []
    representations = collect_all_representations()
    for item in representations:
        representation = item["representation"]
        seed = item["seed"]
        source_x = np.asarray(item["source"]["x"], dtype=np.float32)
        target_x = np.asarray(item["target"]["x"], dtype=np.float32)
        source_y = np.asarray(item["source"]["meta"]["label_binary"], dtype=np.int64)
        target_y = np.asarray(item["target"]["meta"]["label_binary"], dtype=np.int64)

        for condition_idx, condition in enumerate(("overall", "healthy", "adventitious")):
            metrics = compute_condition_metrics(
                source_x=source_x,
                source_y=source_y,
                target_x=target_x,
                target_y=target_y,
                condition=condition,
                seed_for_sampling=ANALYSIS_SEED + seed * 100 + condition_idx,
            )
            rows.append(
                {
                    "representation": representation,
                    "seed": seed,
                    "comparison": "icbhi_train_vs_sprsound_test",
                    "condition": condition,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["representation", "comparison", "condition"], as_index=False)
        .agg(
            rbf_mmd_mean=("rbf_mmd", "mean"),
            rbf_mmd_std=("rbf_mmd", "std"),
            sliced_wasserstein_mean=("sliced_wasserstein", "mean"),
            sliced_wasserstein_std=("sliced_wasserstein", "std"),
            domain_centroid_distance_mean=("domain_centroid_distance", "mean"),
            domain_centroid_distance_std=("domain_centroid_distance", "std"),
            domain_silhouette_mean=("domain_silhouette", "mean"),
            class_silhouette_mean=("class_silhouette", "mean"),
            n_source_mean=("n_source", "mean"),
            n_target_mean=("n_target", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )
    for column in summary.columns:
        if column.endswith("_std"):
            summary[column] = summary[column].fillna(0.0)
    return summary


def build_paper_figure(summary_df: pd.DataFrame) -> None:
    plot_df = summary_df.copy()
    name_map = {
        "cnn_penultimate": "CNN penultimate latent",
        "yamnet_frozen_embedding": "YAMNet frozen embedding",
        "yamnet_trainable_head": "YAMNet trainable-head latent",
    }
    condition_map = {
        "overall": "Overall",
        "healthy": "Healthy only",
        "adventitious": "Adventitious only",
    }
    plot_df["representation_display"] = plot_df["representation"].map(name_map)
    plot_df["condition_display"] = plot_df["condition"].map(condition_map)
    order = [
        "CNN penultimate latent",
        "YAMNet frozen embedding",
        "YAMNet trainable-head latent",
    ]
    cond_order = ["Overall", "Healthy only", "Adventitious only"]

    palette = sns.color_palette("crest", n_colors=len(order))
    color_map = dict(zip(order, palette))
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)

    overall = plot_df[plot_df["condition_display"] == "Overall"].set_index("representation_display").reindex(order)
    healthy = plot_df[plot_df["condition_display"] == "Healthy only"].set_index("representation_display").reindex(order)
    adventitious = plot_df[plot_df["condition_display"] == "Adventitious only"].set_index("representation_display").reindex(order)
    class_conditional = pd.concat(
        [
            healthy.reset_index().assign(condition_display="Healthy only"),
            adventitious.reset_index().assign(condition_display="Adventitious only"),
        ],
        ignore_index=True,
    )

    def _bar_with_errors(ax, values_df, value_col, error_col, title, ylabel):
        ax.bar(
            np.arange(len(order)),
            values_df[value_col].values,
            color=[color_map[name] for name in order],
            edgecolor="black",
            linewidth=0.5,
        )
        ax.errorbar(
            np.arange(len(order)),
            values_df[value_col].values,
            yerr=values_df[error_col].values,
            fmt="none",
            ecolor="black",
            capsize=4,
            linewidth=1.2,
        )
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(order, rotation=14)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")

    _bar_with_errors(
        axes[0, 0],
        overall,
        "rbf_mmd_mean",
        "rbf_mmd_std",
        "Overall inter-domain distance",
        "RBF MMD",
    )
    _bar_with_errors(
        axes[0, 1],
        overall,
        "sliced_wasserstein_mean",
        "sliced_wasserstein_std",
        "Overall inter-domain distance",
        "Sliced Wasserstein",
    )

    for ax, value_col, error_col, title in (
        (axes[1, 0], "rbf_mmd_mean", "rbf_mmd_std", "Class-conditional distance"),
        (axes[1, 1], "sliced_wasserstein_mean", "sliced_wasserstein_std", "Class-conditional distance"),
    ):
        x_positions = np.arange(2)
        width = 0.24
        for idx, rep_name in enumerate(order):
            subset = class_conditional[class_conditional["representation_display"] == rep_name].set_index("condition_display").reindex(cond_order[1:])
            ax.bar(
                x_positions + (idx - 1) * width,
                subset[value_col].values,
                width=width,
                label=rep_name,
                color=color_map[rep_name],
                edgecolor="black",
                linewidth=0.4,
            )
            ax.errorbar(
                x_positions + (idx - 1) * width,
                subset[value_col].values,
                yerr=subset[error_col].values,
                fmt="none",
                ecolor="black",
                capsize=3,
                linewidth=1.0,
            )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(cond_order[1:])
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("RBF MMD" if value_col == "rbf_mmd_mean" else "Sliced Wasserstein")
    axes[1, 1].legend(title="", frameon=False, loc="upper right")

    figure_base = OUTPUT_DIR / "figures" / "domain_distance_icbhi_to_sprsound"
    fig.savefig(f"{figure_base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{figure_base}.pdf", bbox_inches="tight")
    fig.savefig(PAPER_FIGURES_DIR / "figure5_domain_distance_quantification.png", dpi=300, bbox_inches="tight")
    fig.savefig(PAPER_FIGURES_DIR / "figure5_domain_distance_quantification.pdf", bbox_inches="tight")
    plt.close(fig)


def write_readme(summary_df: pd.DataFrame) -> None:
    lines = [
        "# Domain Distance Quantification",
        "",
        "This analysis quantifies inter-domain distance for the main adult-to-pediatric transfer setting (`ICBHI train -> SPRSound test`).",
        "",
        "Representations:",
        "- `cnn_penultimate`: 128-d penultimate latent from the confirmed CNN baseline.",
        "- `yamnet_frozen_embedding`: 1024-d frozen YAMNet mean-pooled embedding.",
        "- `yamnet_trainable_head`: 64-d latent from the confirmed YAMNet trainable-head model.",
        "",
        "Method summary:",
        "- up to 1500 samples per domain and condition",
        "- feature standardization followed by PCA to at most 32 components",
        "- RBF MMD with median-heuristic bandwidth",
        "- sliced Wasserstein distance with 128 random projections",
        "- overall, healthy-only, and adventitious-only comparisons",
        "",
        "Headline numbers:",
    ]
    overall = summary_df[summary_df["condition"] == "overall"].copy()
    for _, row in overall.sort_values("rbf_mmd_mean").iterrows():
        lines.append(
            f"- {row['representation']}: RBF MMD={row['rbf_mmd_mean']:.4f} +- {row['rbf_mmd_std']:.4f}; "
            f"sliced Wasserstein={row['sliced_wasserstein_mean']:.4f} +- {row['sliced_wasserstein_std']:.4f}; "
            f"domain silhouette={row['domain_silhouette_mean']:.4f}; class silhouette={row['class_silhouette_mean']:.4f}"
        )
    (OUTPUT_DIR / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_output_dirs()
    raw_df = quantify_domain_distance()
    raw_csv = OUTPUT_DIR / "tables" / "domain_distance_raw.csv"
    raw_df.to_csv(raw_csv, index=False)

    summary_df = aggregate_metrics(raw_df)
    summary_csv = OUTPUT_DIR / "tables" / "domain_distance_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    build_paper_figure(summary_df)
    write_readme(summary_df)
    print(f">>> Raw distances saved to: {raw_csv}")
    print(f">>> Summary distances saved to: {summary_csv}")
    print(f">>> Paper figure saved to: {PAPER_FIGURES_DIR / 'figure5_domain_distance_quantification.pdf'}")


if __name__ == "__main__":
    main()
