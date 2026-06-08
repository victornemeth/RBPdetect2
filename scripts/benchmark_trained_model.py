#!/usr/bin/env python3
"""Benchmark the trained ESM2 linear classifier on its held-out test set.

Consumes the artifacts written by notebooks/train_classifier.ipynb
(trained_model_test_predictions.tsv) and reports the same metrics as the frozen
embedding benchmark: per-class precision/recall/F1, macro-F1 with a
cluster-bootstrap 95% CI, balanced accuracy, MCC, and a confusion matrix.

The test proteins were never seen during training, so this is a leakage-free
evaluation of the saved model.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

from rbpdetect2.benchmarking import (
    LABELS,
    classification_metrics,
    cluster_bootstrap_macro_f1,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmark_embeddings" / "results"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path,
                    default=RESULTS / "trained_model_test_predictions.tsv")
    ap.add_argument("--checkpoint", type=Path,
                    default=ROOT / "models" / "rbpdetect2_linear_facebook_esm2_t33_650M_UR50D.pt")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.predictions.exists():
        raise SystemExit(
            f"Missing {args.predictions}. Run notebooks/train_classifier.ipynb first "
            "(it writes the held-out test predictions)."
        )

    pred = pd.read_csv(args.predictions, sep="\t")
    y_true = pred["label"].to_numpy()
    y_pred = pred["prediction"].to_numpy()
    clusters = pred["cluster"].to_numpy()

    print(f"Trained-model held-out test set: {len(pred)} proteins")
    print(pred["label"].value_counts().to_string())
    print("\n" + classification_report(y_true, y_pred, labels=LABELS, zero_division=0))

    metrics = classification_metrics(y_true, y_pred)
    ci_low, ci_high = cluster_bootstrap_macro_f1(
        y_true, y_pred, clusters, n_replicates=args.bootstrap, seed=args.seed
    )
    metrics["macro_f1_ci_low"] = ci_low
    metrics["macro_f1_ci_high"] = ci_high
    metrics["n_test"] = len(pred)
    metrics["model"] = "ESM2-650M + linear (trained)"

    row = pd.DataFrame([metrics])
    front = ["model", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high",
             "balanced_accuracy", "accuracy", "mcc"]
    row = row[front + [c for c in row.columns if c not in front]]
    out_csv = RESULTS / "trained_model_metrics.csv"
    row.to_csv(out_csv, index=False)

    print("\n=== summary ===")
    print(f"macro-F1      {metrics['macro_f1']:.4f}  (95% CI {ci_low:.4f}–{ci_high:.4f})")
    print(f"balanced acc  {metrics['balanced_accuracy']:.4f}")
    print(f"accuracy      {metrics['accuracy']:.4f}")
    print(f"MCC           {metrics['mcc']:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, colorbar=False)
    ax.set_title("Trained ESM2 classifier — held-out test")
    fig.tight_layout()
    cm_path = RESULTS / "trained_model_confusion_matrix.png"
    fig.savefig(cm_path, dpi=200, bbox_inches="tight")

    print(f"\nwrote {out_csv}")
    print(f"wrote {cm_path}")

    # If the embedding-probe benchmark exists, show the comparison.
    probe_csv = RESULTS / "linear_probe_metrics.csv"
    if probe_csv.exists():
        probe = pd.read_csv(probe_csv)[["model", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high"]]
        combined = pd.concat([probe, row[["model", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high"]]],
                             ignore_index=True).sort_values("macro_f1", ascending=False)
        print("\n=== vs frozen-embedding probes (NOTE: different test split) ===")
        print(combined.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
