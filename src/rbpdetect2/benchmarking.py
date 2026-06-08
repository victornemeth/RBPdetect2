"""Utilities for reproducible frozen-embedding linear-probe benchmarks."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    silhouette_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rbpdetect2.benchmark_data import dataset_sha256, load_labeled_sequences, sequence_sha256


LABELS = ["TF", "TSP", "nonRBP"]


def load_embedding_bundle(bundle_dir: Path) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Load and validate one extractor output directory."""
    embeddings_path = bundle_dir / "embeddings.npy"
    ids_path = bundle_dir / "ids.tsv"
    metadata_path = bundle_dir / "metadata.json"
    for path in (embeddings_path, ids_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing embedding bundle file: {path}")

    embeddings = np.load(embeddings_path, allow_pickle=False)
    ids = pd.read_csv(ids_path, sep="\t")
    metadata = json.loads(metadata_path.read_text())

    if embeddings.ndim != 2:
        raise ValueError(f"{embeddings_path} must be a 2D array")
    if len(ids) != len(embeddings):
        raise ValueError(f"{ids_path} has {len(ids)} rows but embeddings has {len(embeddings)} rows")
    if ids["id"].duplicated().any():
        duplicate = ids.loc[ids["id"].duplicated(), "id"].iloc[0]
        raise ValueError(f"Duplicate ID in {ids_path}: {duplicate}")
    if metadata["n_sequences"] != len(ids):
        raise ValueError(f"metadata.json n_sequences does not match {ids_path}")
    if metadata["embedding_dim"] != embeddings.shape[1]:
        raise ValueError(f"metadata.json embedding_dim does not match {embeddings_path}")

    return embeddings, ids, metadata


def load_or_create_split(
    *,
    data_dir: Path,
    split_path: Path,
    min_seq_id: float = 0.3,
    coverage: float = 0.8,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Load a committed split, or create an all-sequence MMseqs2 cluster-aware split."""
    records = load_labeled_sequences(data_dir)
    expected_ids = {record["id"] for record in records}
    if split_path.exists():
        split = pd.read_csv(split_path, sep="\t")
        required_columns = {"id", "label", "sequence_sha256", "cluster", "split"}
        if not required_columns.issubset(split.columns):
            missing = sorted(required_columns - set(split.columns))
            raise ValueError(f"{split_path} is missing required columns: {', '.join(missing)}")
        if set(split["id"]) != expected_ids:
            raise ValueError(f"{split_path} IDs do not match the current processed FASTAs")
        expected_hashes = {
            record["id"]: sequence_sha256(record["sequence"])
            for record in records
        }
        observed_hashes = split.set_index("id")["sequence_sha256"].to_dict()
        if observed_hashes != expected_hashes:
            raise ValueError(f"{split_path} sequence hashes do not match the current processed FASTAs")
        return split

    if not shutil.which("mmseqs"):
        raise RuntimeError("mmseqs not found in PATH; it is required to create the benchmark split")

    member_to_rep = _mmseqs_cluster_records(records, min_seq_id=min_seq_id, coverage=coverage)
    split = _cluster_aware_split(
        records,
        member_to_rep,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(split_path, sep="\t", index=False)
    metadata = {
        "schema_version": 1,
        "dataset_sha256": dataset_sha256(records),
        "min_seq_id": min_seq_id,
        "coverage": coverage,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "seed": seed,
    }
    split_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return split


def validate_bundle_against_split(ids: pd.DataFrame, split: pd.DataFrame, bundle_name: str) -> None:
    """Ensure an embedding bundle and benchmark split describe the same labeled proteins."""
    bundle_pairs = set(zip(ids["id"], ids["label"], ids["sequence_sha256"], strict=True))
    split_pairs = set(zip(split["id"], split["label"], split["sequence_sha256"], strict=True))
    if bundle_pairs != split_pairs:
        raise ValueError(f"Embedding bundle {bundle_name} does not match the benchmark split")


def run_linear_probe(
    *,
    embeddings: np.ndarray,
    ids: pd.DataFrame,
    split: pd.DataFrame,
    c_values: list[float],
    max_iter: int = 10_000,
    bootstrap_replicates: int = 2_000,
    bootstrap_seed: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Tune a balanced multinomial logistic regression and evaluate the held-out test set."""
    frame = ids[["id", "label"]].merge(
        split[["id", "cluster", "split"]],
        on="id",
        validate="one_to_one",
    )
    row_lookup = pd.Series(np.arange(len(ids)), index=ids["id"])
    indices = row_lookup.loc[frame["id"]].to_numpy()
    X = embeddings[indices]
    y = frame["label"].to_numpy()

    train_mask = frame["split"].eq("train").to_numpy()
    val_mask = frame["split"].eq("val").to_numpy()
    test_mask = frame["split"].eq("test").to_numpy()

    validation_scores: dict[float, float] = {}
    for c_value in c_values:
        probe = _probe_pipeline(c_value, max_iter=max_iter)
        probe.fit(X[train_mask], y[train_mask])
        val_predictions = probe.predict(X[val_mask])
        validation_scores[c_value] = f1_score(
            y[val_mask],
            val_predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )

    best_c = max(c_values, key=lambda value: (validation_scores[value], -value))
    fit_mask = train_mask | val_mask
    probe = _probe_pipeline(best_c, max_iter=max_iter)
    probe.fit(X[fit_mask], y[fit_mask])
    predictions = probe.predict(X[test_mask])
    probabilities = probe.predict_proba(X[test_mask])
    test_labels = y[test_mask]
    test_clusters = frame.loc[test_mask, "cluster"].to_numpy()

    metrics = classification_metrics(test_labels, predictions)
    ci_low, ci_high = cluster_bootstrap_macro_f1(
        test_labels,
        predictions,
        test_clusters,
        n_replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    metrics.update(
        {
            "best_c": best_c,
            "validation_macro_f1": validation_scores[best_c],
            "macro_f1_ci_low": ci_low,
            "macro_f1_ci_high": ci_high,
            "validation_scores": validation_scores,
        }
    )

    test_frame = frame.loc[test_mask, ["id", "label", "cluster", "split"]].copy()
    test_frame["prediction"] = predictions
    for class_index, label in enumerate(probe.named_steps["classifier"].classes_):
        test_frame[f"probability_{label}"] = probabilities[:, class_index]
    return metrics, test_frame


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
    )
    metrics: dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    for index, label in enumerate(LABELS):
        metrics[f"{label}_precision"] = precision[index]
        metrics[f"{label}_recall"] = recall[index]
        metrics[f"{label}_f1"] = f1[index]
        metrics[f"{label}_support"] = int(support[index])
    return metrics


def cluster_bootstrap_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    clusters: np.ndarray,
    *,
    n_replicates: int,
    seed: int,
) -> tuple[float, float]:
    """Calculate a percentile CI while resampling test clusters, not individual proteins."""
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(clusters)
    cluster_indices = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique_clusters}
    scores = np.empty(n_replicates, dtype=float)

    for replicate in range(n_replicates):
        sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        sampled_indices = np.concatenate([cluster_indices[cluster] for cluster in sampled_clusters])
        scores[replicate] = f1_score(
            y_true[sampled_indices],
            y_pred[sampled_indices],
            labels=LABELS,
            average="macro",
            zero_division=0,
        )
    return tuple(np.quantile(scores, [0.025, 0.975]))


def embedding_separation_metrics(
    *,
    embeddings: np.ndarray,
    ids: pd.DataFrame,
    split: pd.DataFrame,
    n_neighbors: int = 5,
) -> dict[str, float]:
    """Calculate descriptive high-dimensional separation metrics."""
    frame = ids[["id", "label"]].merge(split[["id", "split"]], on="id", validate="one_to_one")
    row_lookup = pd.Series(np.arange(len(ids)), index=ids["id"])
    X = embeddings[row_lookup.loc[frame["id"]].to_numpy()]
    y = frame["label"].to_numpy()
    train_mask = frame["split"].eq("train").to_numpy()
    test_mask = frame["split"].eq("test").to_numpy()

    test_silhouette = silhouette_score(X[test_mask], y[test_mask], metric="cosine")
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="cosine", weights="distance")
    knn.fit(X[train_mask], y[train_mask])
    knn_predictions = knn.predict(X[test_mask])
    return {
        "cosine_silhouette": test_silhouette,
        f"{n_neighbors}nn_balanced_accuracy": balanced_accuracy_score(y[test_mask], knn_predictions),
    }


def _probe_pipeline(c_value: float, *, max_iter: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=max_iter,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _mmseqs_cluster_records(
    records: list[dict[str, str]],
    *,
    min_seq_id: float,
    coverage: float,
) -> dict[str, str]:
    with tempfile.TemporaryDirectory() as tmp_root:
        tmp = Path(tmp_root)
        fasta = tmp / "all_sequences.fasta"
        with fasta.open("w") as handle:
            for record in records:
                handle.write(f">{record['id']}\n{record['sequence']}\n")

        prefix = tmp / "clusters"
        subprocess.run(
            [
                "mmseqs",
                "easy-cluster",
                str(fasta),
                str(prefix),
                str(tmp / "work"),
                "--min-seq-id",
                str(min_seq_id),
                "-c",
                str(coverage),
                "--cov-mode",
                "0",
                "--cluster-mode",
                "2",
                "-v",
                "0",
            ],
            check=True,
        )
        member_to_rep: dict[str, str] = {}
        with Path(f"{prefix}_cluster.tsv").open() as handle:
            for line in handle:
                representative, member = line.rstrip().split("\t")
                member_to_rep[member] = representative
    return member_to_rep


def _cluster_aware_split(
    records: list[dict[str, str]],
    member_to_rep: dict[str, str],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "id": [record["id"] for record in records],
            "label": [record["label"] for record in records],
            "sequence_length": [len(record["sequence"]) for record in records],
            "sequence_sha256": [sequence_sha256(record["sequence"]) for record in records],
        }
    )
    frame["cluster"] = frame["id"].map(member_to_rep)
    if frame["cluster"].isna().any():
        missing = frame.loc[frame["cluster"].isna(), "id"].iloc[0]
        raise ValueError(f"MMseqs2 did not assign a cluster to {missing}")

    cluster_info = (
        frame.groupby("cluster")["label"]
        .agg(lambda labels: labels.value_counts().index[0])
        .rename("dominant_label")
        .reset_index()
    )
    first_split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_fraction + test_fraction,
        random_state=seed,
    )
    train_index, holdout_index = next(
        first_split.split(cluster_info["cluster"], cluster_info["dominant_label"])
    )
    holdout = cluster_info.iloc[holdout_index]
    second_split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_fraction / (val_fraction + test_fraction),
        random_state=seed,
    )
    val_index, test_index = next(
        second_split.split(holdout["cluster"], holdout["dominant_label"])
    )

    assignments = {
        **{cluster: "train" for cluster in cluster_info.iloc[train_index]["cluster"]},
        **{cluster: "val" for cluster in holdout.iloc[val_index]["cluster"]},
        **{cluster: "test" for cluster in holdout.iloc[test_index]["cluster"]},
    }
    frame["split"] = frame["cluster"].map(assignments)
    return frame

