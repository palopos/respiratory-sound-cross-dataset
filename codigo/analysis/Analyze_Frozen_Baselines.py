from __future__ import annotations

import os
import re
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
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    import umap
except Exception:
    umap = None


OUTPUT_DIR = Path("./modelos/comparativas/analisis_frozen")
COMPARISON_LONG_CSV = Path("./modelos/comparativas/comparativa_modelos_frozen_test_long.csv")
SPLIT_MANIFESTS = {
    "sprsound": Path("./ficheros/splits/0 SPRSound/splits_patient_independent_seed42.csv"),
    "icbhi": Path("./ficheros/splits/0 ICBHI/splits_patient_independent_seed42.csv"),
}
PURE_CROSS_PROTOCOLS = {"icbhi_to_sprsound", "sprsound_to_icbhi"}
EMBEDDING_INPUTS = {
    "yamnet": {
        "sprsound": Path("./ficheros/3 Embeddings YAMNet/0 SPRSound_seed42_yamnet_tfhub_mean_pool_1024"),
        "icbhi": Path("./ficheros/3 Embeddings YAMNet/0 ICBHI_seed42_yamnet_tfhub_mean_pool_1024"),
    },
    "wav2vec2": {
        "sprsound": Path("./ficheros/3 Embeddings wav2vec2/0 SPRSound_seed42_wav2vec2_hf_mean_pool_768"),
        "icbhi": Path("./ficheros/3 Embeddings wav2vec2/0 ICBHI_seed42_wav2vec2_hf_mean_pool_768"),
    },
}
SEGMENT_STATS_INPUTS = {
    "sprsound": Path("./ficheros/3 Embeddings wav2vec2/0 SPRSound_seed42_wav2vec2_hf_mean_pool_768"),
    "icbhi": Path("./ficheros/3 Embeddings wav2vec2/0 ICBHI_seed42_wav2vec2_hf_mean_pool_768"),
    "fusion": Path("./ficheros/3 Embeddings wav2vec2/0 Fusion SPRSound ICBHI_seed42_wav2vec2_hf_mean_pool_768"),
}


def ensure_output_dirs() -> None:
    for subdir in ("figures", "tables"):
        (OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)


def slugify(value: str) -> str:
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


def parse_protocol(protocol: str) -> tuple[str, str]:
    if "_to_" not in protocol:
        raise ValueError(f"Protocolo no soportado: {protocol}")
    return tuple(protocol.split("_to_", 1))


def infer_label_name_from_path(sample_path: str) -> str:
    return Path(sample_path).parent.name


def infer_source_dataset_from_path(sample_path: str) -> str:
    sample_path = str(sample_path)
    if (
        "/0 SPRSound/" in sample_path
        or "/0 SPRSound_seed" in sample_path
        or sample_path.startswith("./ficheros/0 SPRSound")
    ):
        return "sprsound"
    if (
        "/0 ICBHI/" in sample_path
        or "/0 ICBHI_seed" in sample_path
        or sample_path.startswith("./ficheros/0 ICBHI")
    ):
        return "icbhi"
    if (
        "/0 Fusion SPRSound ICBHI/" in sample_path
        or "/0 Fusion SPRSound ICBHI_seed" in sample_path
        or sample_path.startswith("./ficheros/0 Fusion SPRSound ICBHI")
    ):
        return "sprsound_icbhi_fusion"
    if "sprsound__" in sample_path:
        return "sprsound"
    if "icbhi__" in sample_path:
        return "icbhi"
    return "unknown"


def infer_recording_id(segment_file: str) -> str:
    stem = Path(segment_file).stem
    return re.sub(r"_ev\d+$", "", stem)


def load_split_manifests() -> dict[str, pd.DataFrame]:
    manifests = {}
    for dataset_name, path in SPLIT_MANIFESTS.items():
        df = pd.read_csv(path)
        df["dataset_name"] = dataset_name
        manifests[dataset_name] = df
    return manifests


def collect_prediction_files() -> list[dict]:
    compare_df = pd.read_csv(COMPARISON_LONG_CSV)
    rows = []
    for _, row in compare_df.iterrows():
        source_csv = Path(row["source_csv"])
        run_dir = source_csv.parents[1]
        for prediction_csv in sorted(run_dir.glob("seed_*/predictions/predicciones_test.csv")):
            seed_dir = prediction_csv.parents[1]
            rows.append(
                {
                    "experiment_block": row["experiment_block"],
                    "dataset_protocol": row["dataset_protocol"],
                    "model_family": row["model_family"],
                    "run_id": row["run_id"],
                    "run_dir": run_dir,
                    "seed": int(seed_dir.name.replace("seed_", "")),
                    "prediction_csv": prediction_csv,
                }
            )
    return rows


def enrich_predictions(df: pd.DataFrame, protocol: str, model_family: str) -> pd.DataFrame:
    df = df.copy()
    if "segment_file" not in df.columns:
        df["segment_file"] = df["sample_path"].map(lambda value: Path(value).name)
    if "label_name" not in df.columns:
        df["label_name"] = df["sample_path"].map(infer_label_name_from_path)
    if "source_dataset" not in df.columns:
        df["source_dataset"] = df["sample_path"].map(infer_source_dataset_from_path)
    df["recording_id"] = df["segment_file"].map(infer_recording_id)
    df["model_family"] = model_family
    df["dataset_protocol"] = protocol
    train_dataset, test_dataset = parse_protocol(protocol)
    df["train_dataset"] = train_dataset
    df["test_dataset"] = test_dataset
    df["predicted_positive"] = df["y_pred_threshold_0_5"].astype(int)
    df["is_error"] = (df["predicted_positive"] != df["y_true"]).astype(int)
    df["error_type"] = np.select(
        [
            (df["y_true"] == 0) & (df["predicted_positive"] == 0),
            (df["y_true"] == 0) & (df["predicted_positive"] == 1),
            (df["y_true"] == 1) & (df["predicted_positive"] == 0),
            (df["y_true"] == 1) & (df["predicted_positive"] == 1),
        ],
        ["TN", "FP", "FN", "TP"],
        default="UNKNOWN",
    )
    df["confidence"] = np.where(
        df["predicted_positive"] == 1,
        df["prob_adventitious"],
        1.0 - df["prob_adventitious"],
    )
    df["prob_adventitious_inverted"] = 1.0 - df["prob_adventitious"]
    return df


def attach_manifest_metadata(df: pd.DataFrame, manifests: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for dataset_name, sub_df in df.groupby("source_dataset", dropna=False):
        if dataset_name in manifests:
            merged = sub_df.merge(
                manifests[dataset_name],
                on="recording_id",
                how="left",
                suffixes=("", "_manifest"),
            )
        else:
            merged = sub_df.copy()
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def load_all_predictions() -> pd.DataFrame:
    manifests = load_split_manifests()
    frames = []
    for record in collect_prediction_files():
        df = pd.read_csv(record["prediction_csv"])
        df = enrich_predictions(df, record["dataset_protocol"], record["model_family"])
        df["experiment_block"] = record["experiment_block"]
        df["run_id"] = record["run_id"]
        df["prediction_csv"] = str(record["prediction_csv"])
        df["seed"] = record["seed"]
        frames.append(df)
    all_predictions = pd.concat(frames, ignore_index=True)
    return attach_manifest_metadata(all_predictions, manifests)


def summarize_transfer_behavior(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    transfer_df = predictions[predictions["train_dataset"] != predictions["test_dataset"]].copy()
    grouped = transfer_df.groupby(["dataset_protocol", "model_family", "experiment_block", "run_id", "seed"], sort=False)
    for keys, df in grouped:
        raw_auc = roc_auc_score(df["y_true"], df["prob_adventitious"])
        inverted_auc = roc_auc_score(df["y_true"], df["prob_adventitious_inverted"])
        raw_pr = average_precision_score(df["y_true"], df["prob_adventitious"])
        inverted_pr = average_precision_score(df["y_true"], df["prob_adventitious_inverted"])
        tn, fp, fn, tp = confusion_matrix(df["y_true"], df["predicted_positive"], labels=[0, 1]).ravel()
        rows.append(
            {
                "dataset_protocol": keys[0],
                "model_family": keys[1],
                "experiment_block": keys[2],
                "run_id": keys[3],
                "seed": keys[4],
                "n_segments": len(df),
                "n_recordings": df["recording_id"].nunique(),
                "raw_auc_roc": raw_auc,
                "inverted_auc_roc": inverted_auc,
                "auc_gain_if_inverted": inverted_auc - raw_auc,
                "raw_auc_pr": raw_pr,
                "inverted_auc_pr": inverted_pr,
                "predicted_positive_rate": df["predicted_positive"].mean(),
                "true_positive_rate_dataset": df["y_true"].mean(),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "recall_adv": tp / (tp + fn + 1e-9),
                "specificity": tn / (tn + fp + 1e-9),
                "precision_adv": tp / (tp + fp + 1e-9),
                "error_rate": df["is_error"].mean(),
            }
        )
    summary_df = pd.DataFrame(rows).sort_values(["dataset_protocol", "model_family", "seed"])
    summary_df.to_csv(OUTPUT_DIR / "tables" / "transfer_behavior_by_seed.csv", index=False)
    aggregate = summary_df.groupby(["dataset_protocol", "model_family"], sort=False).agg(
        n_seeds=("seed", "nunique"),
        raw_auc_roc_mean=("raw_auc_roc", "mean"),
        raw_auc_roc_std=("raw_auc_roc", "std"),
        inverted_auc_roc_mean=("inverted_auc_roc", "mean"),
        auc_gain_if_inverted_mean=("auc_gain_if_inverted", "mean"),
        raw_auc_pr_mean=("raw_auc_pr", "mean"),
        predicted_positive_rate_mean=("predicted_positive_rate", "mean"),
        error_rate_mean=("error_rate", "mean"),
        recall_adv_mean=("recall_adv", "mean"),
        specificity_mean=("specificity", "mean"),
        precision_adv_mean=("precision_adv", "mean"),
    ).reset_index()
    aggregate.to_csv(OUTPUT_DIR / "tables" / "transfer_behavior_aggregate.csv", index=False)
    return aggregate


def summarize_errors_by_label(predictions: pd.DataFrame) -> None:
    transfer_df = predictions[predictions["train_dataset"] != predictions["test_dataset"]].copy()
    by_label = transfer_df.groupby(
        ["dataset_protocol", "model_family", "seed", "label_name"],
        sort=False,
    ).agg(
        n_segments=("label_name", "size"),
        n_errors=("is_error", "sum"),
        error_rate=("is_error", "mean"),
        predicted_positive_rate=("predicted_positive", "mean"),
        n_fp=("error_type", lambda values: int(np.sum(values == "FP"))),
        n_fn=("error_type", lambda values: int(np.sum(values == "FN"))),
        n_tp=("error_type", lambda values: int(np.sum(values == "TP"))),
        n_tn=("error_type", lambda values: int(np.sum(values == "TN"))),
    ).reset_index()
    by_label.to_csv(OUTPUT_DIR / "tables" / "transfer_errors_by_label.csv", index=False)


def summarize_errors_by_metadata(predictions: pd.DataFrame) -> None:
    transfer_df = predictions[predictions["train_dataset"] != predictions["test_dataset"]].copy()
    metadata_columns = [
        "record_annotation",
        "location",
        "chest_location",
        "acquisition_mode",
        "equipment",
        "gender",
    ]
    rows = []
    for column in metadata_columns:
        if column not in transfer_df.columns:
            continue
        sub_df = transfer_df[transfer_df[column].notna()].copy()
        grouped = sub_df.groupby(["dataset_protocol", "model_family", "seed", "source_dataset", column], sort=False)
        for keys, group in grouped:
            rows.append(
                {
                    "dataset_protocol": keys[0],
                    "model_family": keys[1],
                    "seed": keys[2],
                    "source_dataset": keys[3],
                    "metadata_field": column,
                    "metadata_value": keys[4],
                    "n_segments": len(group),
                    "n_recordings": group["recording_id"].nunique(),
                    "error_rate": group["is_error"].mean(),
                    "predicted_positive_rate": group["predicted_positive"].mean(),
                    "positive_rate_true": group["y_true"].mean(),
                }
            )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "tables" / "transfer_errors_by_metadata.csv", index=False)


def export_top_confident_errors(predictions: pd.DataFrame, top_k: int = 40) -> None:
    transfer_df = predictions[predictions["train_dataset"] != predictions["test_dataset"]].copy()
    misclassified = transfer_df[transfer_df["is_error"] == 1].copy()
    columns = [
        "dataset_protocol", "model_family", "seed", "error_type", "confidence",
        "prob_adventitious", "y_true", "predicted_positive", "label_name",
        "recording_id", "sample_path", "source_dataset", "record_annotation",
        "location", "chest_location", "acquisition_mode", "equipment",
    ]
    frames = []
    for keys, group in misclassified.groupby(["dataset_protocol", "model_family", "seed"], sort=False):
        available_columns = [column for column in columns if column in group.columns]
        frames.append(group.sort_values("confidence", ascending=False).head(top_k)[available_columns])
    pd.concat(frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "tables" / "transfer_top_confident_errors.csv",
        index=False,
    )


def load_metadata_table(base_dir: Path, split_name: str) -> pd.DataFrame:
    split_file = base_dir / f"segmentado_{split_name}_metadata.csv"
    df = pd.read_csv(split_file)
    df["container_dataset"] = base_dir.name
    df["split"] = split_name
    return df


def build_segment_statistics() -> None:
    frames = []
    for dataset_key, base_dir in SEGMENT_STATS_INPUTS.items():
        for split_name in ("train", "validation", "test"):
            df = load_metadata_table(base_dir, split_name)
            df["dataset_key"] = dataset_key
            frames.append(df)
    all_meta = pd.concat(frames, ignore_index=True)

    summary = all_meta.groupby(["dataset_key", "split"], sort=False).agg(
        n_segments=("sample_path", "size"),
        n_recordings=("sample_path", lambda values: len({infer_recording_id(Path(value).name) for value in values})),
        positive_rate=("label_binary", "mean"),
        duration_mean=("duration_seconds_resampled", "mean"),
        duration_std=("duration_seconds_resampled", "std"),
        duration_median=("duration_seconds_resampled", "median"),
        duration_p10=("duration_seconds_resampled", lambda values: float(np.quantile(values, 0.10))),
        duration_p90=("duration_seconds_resampled", lambda values: float(np.quantile(values, 0.90))),
    ).reset_index()
    summary.to_csv(OUTPUT_DIR / "tables" / "segment_duration_stats_by_dataset_split.csv", index=False)

    by_label = all_meta.groupby(["dataset_key", "split", "label_name"], sort=False).agg(
        n_segments=("label_name", "size"),
        duration_mean=("duration_seconds_resampled", "mean"),
        duration_median=("duration_seconds_resampled", "median"),
    ).reset_index()
    by_label.to_csv(OUTPUT_DIR / "tables" / "segment_label_distribution_by_dataset_split.csv", index=False)

    test_only = all_meta[all_meta["split"] == "test"].copy()
    comparisons = []
    for label_value, label_name in [(None, "all"), (0, "normal"), (1, "adventitious")]:
        if label_value is None:
            spr = test_only[test_only["dataset_key"] == "sprsound"]["duration_seconds_resampled"]
            icb = test_only[test_only["dataset_key"] == "icbhi"]["duration_seconds_resampled"]
        else:
            spr = test_only[
                (test_only["dataset_key"] == "sprsound") & (test_only["label_binary"] == label_value)
            ]["duration_seconds_resampled"]
            icb = test_only[
                (test_only["dataset_key"] == "icbhi") & (test_only["label_binary"] == label_value)
            ]["duration_seconds_resampled"]
        ks = ks_2samp(spr, icb)
        comparisons.append(
            {
                "subset": label_name,
                "sprsound_n": len(spr),
                "icbhi_n": len(icb),
                "sprsound_duration_mean": spr.mean(),
                "icbhi_duration_mean": icb.mean(),
                "ks_statistic": ks.statistic,
                "ks_pvalue": ks.pvalue,
            }
        )
    pd.DataFrame(comparisons).to_csv(OUTPUT_DIR / "tables" / "duration_shift_sprsound_vs_icbhi_test.csv", index=False)


def _prepare_embedding_table(model_family: str) -> tuple[np.ndarray, pd.DataFrame]:
    frames = []
    embedding_arrays = []
    for dataset_key, base_dir in EMBEDDING_INPUTS[model_family].items():
        npz = np.load(base_dir / "segmentado_test_embeddings.npz", allow_pickle=False)
        embeddings = np.asarray(npz["embeddings"], dtype=np.float32)
        metadata = pd.read_csv(base_dir / "segmentado_test_metadata.csv")
        metadata["dataset_key"] = dataset_key
        if "source_dataset" not in metadata.columns:
            metadata["source_dataset"] = dataset_key
        metadata["embedding_model_family"] = model_family
        if len(metadata) != len(embeddings):
            raise ValueError(f"Desajuste en embeddings/metadata para {base_dir}")
        frames.append(metadata)
        embedding_arrays.append(embeddings)
    return np.vstack(embedding_arrays), pd.concat(frames, ignore_index=True)


def _reduce_embeddings(embeddings: np.ndarray) -> tuple[np.ndarray, str]:
    scaled = StandardScaler().fit_transform(embeddings)
    n_components = min(50, scaled.shape[0] - 1, scaled.shape[1])
    if n_components < 2:
        raise ValueError("No hay suficientes muestras para reducir embeddings.")
    pca = PCA(n_components=n_components, random_state=42)
    pca_embeddings = pca.fit_transform(scaled)
    if umap is not None:
        reducer = umap.UMAP(n_components=2, n_neighbors=25, min_dist=0.15, metric="euclidean", random_state=42)
        return reducer.fit_transform(pca_embeddings), "umap"
    return pca_embeddings[:, :2], "pca2"


def _centroid_distance(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    centroid_a = np.mean(a, axis=0)
    centroid_b = np.mean(b, axis=0)
    euclidean = float(np.linalg.norm(centroid_a - centroid_b))
    denom = (np.linalg.norm(centroid_a) * np.linalg.norm(centroid_b)) + 1e-9
    cosine_distance = float(1.0 - np.dot(centroid_a, centroid_b) / denom)
    return euclidean, cosine_distance


def analyze_embedding_shift() -> None:
    summary_rows = []
    for model_family in ("yamnet", "wav2vec2"):
        embeddings, metadata = _prepare_embedding_table(model_family)
        coords, method_name = _reduce_embeddings(embeddings)
        metadata = metadata.copy()
        metadata["coord_1"] = coords[:, 0]
        metadata["coord_2"] = coords[:, 1]
        metadata.to_csv(OUTPUT_DIR / "tables" / f"{model_family}_test_embedding_projection.csv", index=False)

        scaled = StandardScaler().fit_transform(embeddings)
        pca_dim = min(30, scaled.shape[0] - 1, scaled.shape[1])
        pca = PCA(n_components=pca_dim, random_state=42)
        pca_embeddings = pca.fit_transform(scaled)

        dataset_codes = metadata["dataset_key"].astype("category").cat.codes
        class_codes = metadata["label_binary"].astype(int)
        summary_rows.append(
            {
                "model_family": model_family,
                "projection_method": method_name,
                "n_segments": len(metadata),
                "dataset_silhouette": float(silhouette_score(pca_embeddings, dataset_codes)),
                "class_silhouette": float(silhouette_score(pca_embeddings, class_codes)),
            }
        )

        spr = embeddings[metadata["dataset_key"] == "sprsound"]
        icb = embeddings[metadata["dataset_key"] == "icbhi"]
        euclidean, cosine_distance = _centroid_distance(spr, icb)
        summary_rows.append(
            {
                "model_family": model_family,
                "projection_method": "centroids_all",
                "n_segments": len(metadata),
                "dataset_silhouette": np.nan,
                "class_silhouette": np.nan,
                "centroid_euclidean": euclidean,
                "centroid_cosine_distance": cosine_distance,
                "subset": "all",
            }
        )
        for label_binary, subset_name in [(0, "normal"), (1, "adventitious")]:
            subset = metadata["label_binary"] == label_binary
            spr_subset = embeddings[(metadata["dataset_key"] == "sprsound") & subset]
            icb_subset = embeddings[(metadata["dataset_key"] == "icbhi") & subset]
            if len(spr_subset) == 0 or len(icb_subset) == 0:
                continue
            euclidean, cosine_distance = _centroid_distance(spr_subset, icb_subset)
            summary_rows.append(
                {
                    "model_family": model_family,
                    "projection_method": "centroids_by_class",
                    "n_segments": int(np.sum(subset)),
                    "dataset_silhouette": np.nan,
                    "class_silhouette": np.nan,
                    "centroid_euclidean": euclidean,
                    "centroid_cosine_distance": cosine_distance,
                    "subset": subset_name,
                }
            )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.scatterplot(
            data=metadata,
            x="coord_1",
            y="coord_2",
            hue="dataset_key",
            style="dataset_key",
            s=18,
            alpha=0.65,
            ax=axes[0],
        )
        axes[0].set_title(f"{model_family} | dataset")
        axes[0].legend(loc="best", fontsize=8)

        sns.scatterplot(
            data=metadata,
            x="coord_1",
            y="coord_2",
            hue="label_name",
            s=18,
            alpha=0.60,
            ax=axes[1],
        )
        axes[1].set_title(f"{model_family} | original class")
        axes[1].legend(loc="best", fontsize=7)
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "figures" / f"{model_family}_embedding_projection_test.png", dpi=220)
        plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "tables" / "embedding_shift_summary.csv", index=False)


def export_protocol_rankings(predictions: pd.DataFrame) -> None:
    transfer_df = predictions[predictions["train_dataset"] != predictions["test_dataset"]].copy()
    ranking_rows = []
    for (protocol, family), group in transfer_df.groupby(["dataset_protocol", "model_family"], sort=False):
        ranking_rows.append(
            {
                "dataset_protocol": protocol,
                "model_family": family,
                "mean_error_rate": group["is_error"].mean(),
                "mean_predicted_positive_rate": group["predicted_positive"].mean(),
                "mean_true_positive_rate": group["y_true"].mean(),
            }
        )
    pd.DataFrame(ranking_rows).to_csv(OUTPUT_DIR / "tables" / "transfer_prediction_behavior.csv", index=False)


def main() -> None:
    ensure_output_dirs()
    predictions = load_all_predictions()
    predictions.to_csv(OUTPUT_DIR / "tables" / "all_test_predictions_enriched.csv", index=False)
    summarize_transfer_behavior(predictions)
    summarize_errors_by_label(predictions)
    summarize_errors_by_metadata(predictions)
    export_top_confident_errors(predictions)
    export_protocol_rankings(predictions)
    build_segment_statistics()
    analyze_embedding_shift()
    print(f">>> Analisis frozen guardado en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
