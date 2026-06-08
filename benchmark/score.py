#!/usr/bin/env python
"""
Score every tool's predictions against a benchmark ground truth and emit a
side-by-side comparison table.

Usage:
    python score.py inphared        # temporal held-out set (post-Aug-2025 phages)
    python score.py experimental    # experimentally-validated proteins

For each <set>/ folder it reads:
    <set>/data/benchmark.tsv          ground truth (columns: protein_id, label, ...)
    <set>/predictions/<tool>.tsv      one TSV per tool
and writes:
    <set>/results/metrics_overall.tsv
    <set>/results/recall_by_label.tsv
    <set>/results/rbpdetect_confusion_matrix.tsv
    <set>/results/metrics_by_tier.tsv      (experimental set only)
    <set>/results/summary.md

Label space: NEG (negative), TF (tail fiber), TSP (tail spike), RBP (generic
receptor-binding protein). Binary RBP-positive = label != NEG.

To benchmark a newly trained RBPdetect2 model, drop its predictions at
<set>/predictions/rbpdetect_new.tsv (see run_classifier.py). It is picked up
automatically and reported next to the previous model and every other tool.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

HERE = Path(__file__).resolve().parent

# Registry: filename -> (display name, prediction column, task, is_three_class)
#   task "rbp" -> positive = label != NEG ; "tsp" -> positive = label == TSP
#   is_three_class -> also emit a 0/1/2 confusion matrix (RBPdetect2 only)
TOOLS = [
    ("rbpdetect2.tsv",   "RBPdetect2 (previous)",               "pred",            "rbp", True),
    ("rbpdetect_new.tsv","RBPdetect2 (new, this work)",         "pred",            "rbp", True),
    ("rbpdetect_v4.tsv", "PhageRBPdetect v4 (Boeckaerts 2024)", "pred",            "rbp", False),
    ("phold.tsv",        "Phold (Bouras 2024)",                 "pred",            "rbp", False),
    ("pharokka.tsv",     "Pharokka (Bouras 2023)",              "pred",            "rbp", False),
    ("phanns.tsv",       "PhANNs (Cantu 2020)",                 "pred",            "rbp", False),
    ("blastp.tsv",       "BLASTp baseline",                     "pred",            "rbp", False),
    ("deposcope.tsv",    "DepoScope (Concha-Eloko 2024)",       "pred",            "tsp", False),
    ("spikehunter.tsv",  "SpikeHunter (Yang 2024)",             "Predicted_label", "tsp", False),
]


def binary_metrics(y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(set(y_pred)) > 1 else 0.0,
    }


def per_label_recall(name, merged, pred_col) -> dict:
    out = {"tool": name}
    for lbl in ["TF", "TSP", "RBP"]:
        sub = merged[merged["label"] == lbl]
        out[f"recall_{lbl}"] = (sub[pred_col] == 1).mean() if len(sub) else float("nan")
        out[f"n_{lbl}"] = len(sub)
    return out


def main(set_name: str) -> int:
    base = HERE / set_name
    if not base.is_dir():
        sys.exit(f"unknown benchmark set '{set_name}'. Expected a folder at {base}")
    pred_dir = base / "predictions"
    out_dir = base / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    bench = pd.read_csv(base / "data" / "benchmark.tsv", sep="\t",
                        dtype=str, keep_default_na=False)
    bench["y_rbp"] = (bench["label"] != "NEG").astype(int)
    bench["y_tsp"] = (bench["label"] == "TSP").astype(int)
    has_tier = "evidence_tier" in bench.columns
    if has_tier:
        bench["evidence_tier"] = bench["evidence_tier"].astype(int)

    print(f"[{set_name}] benchmark size: {len(bench)}  "
          f"(TF={(bench.label=='TF').sum()}, TSP={(bench.label=='TSP').sum()}, "
          f"RBP={(bench.label=='RBP').sum()}, NEG={(bench.label=='NEG').sum()})")

    rows_overall, rows_recall, rows_tier = [], [], []
    rbp_confusion = None  # last three-class confusion matrix (new model if present)

    for fname, name, col, task, three_class in TOOLS:
        fpath = pred_dir / fname
        if not fpath.exists():
            continue
        pred = pd.read_csv(fpath, sep="\t")
        # Collapse to RBP-positive: != 0 is identity for binary tools and folds
        # the 3-class (0=NEG/1=TF/2=TSP) tools (RBPdetect2, BLASTp) into one positive.
        pred["pred_bin"] = (pred[col].astype(int) != 0).astype(int)
        merged = bench.merge(pred[["protein_id", col, "pred_bin"]], on="protein_id", how="left")
        n_missing = int(merged["pred_bin"].isna().sum())
        if n_missing:
            print(f"  WARNING: {name} missing {n_missing} predictions; treated as 0")
        merged["pred_bin"] = merged["pred_bin"].fillna(0).astype(int)

        y_true = merged.y_tsp if task == "tsp" else merged.y_rbp
        task_label = "TSP-only" if task == "tsp" else "RBP (TF|TSP|RBP)"
        rows_overall.append({"tool": name, "task": task_label, **binary_metrics(y_true, merged.pred_bin)})
        rows_recall.append(per_label_recall(name, merged, "pred_bin"))

        if has_tier:
            for t in sorted(bench.evidence_tier.unique()):
                sub = merged[merged.evidence_tier == t]
                if sub.empty:
                    continue
                yt = sub.y_tsp if task == "tsp" else sub.y_rbp
                rows_tier.append({"tool": name, "tier": int(t), "n": len(sub),
                                  "task": task_label, **binary_metrics(yt, sub.pred_bin)})

        if three_class:
            m = bench.merge(pred[["protein_id", col]], on="protein_id", how="left")
            rbp_confusion = pd.crosstab(m["label"], m[col].fillna(-1).astype(int),
                                        rownames=["true"], colnames=["pred"], margins=True)

    # ---- write + print ----
    df_overall = pd.DataFrame(rows_overall)
    df_overall.to_csv(out_dir / "metrics_overall.tsv", sep="\t", index=False)
    cols = ["tool", "task", "TP", "FP", "TN", "FN", "precision", "recall",
            "specificity", "f1", "mcc", "accuracy"]
    print("\n=== Overall metrics ===")
    print(df_overall[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    df_recall = pd.DataFrame(rows_recall)
    df_recall.to_csv(out_dir / "recall_by_label.tsv", sep="\t", index=False)
    print("\n=== Recall by ground-truth label slice ===")
    print(df_recall[["tool", "recall_TF", "recall_TSP", "recall_RBP"]].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))

    if has_tier and rows_tier:
        df_tier = pd.DataFrame(rows_tier)
        df_tier.to_csv(out_dir / "metrics_by_tier.tsv", sep="\t", index=False)
        print("\n=== By evidence tier ===")
        print(df_tier[["tool", "tier", "n", "precision", "recall", "f1", "mcc"]].to_string(
            index=False, float_format=lambda x: f"{x:.3f}"))

    if rbp_confusion is not None:
        rbp_confusion.to_csv(out_dir / "rbpdetect_confusion_matrix.tsv", sep="\t")
        print("\n=== RBPdetect2 3-class confusion (rows=true, cols=pred 0=NEG/1=TF/2=TSP) ===")
        print(rbp_confusion.to_string())

    # ---- markdown summary ----
    md = [f"# Benchmark results — {set_name}", "",
          f"Ground truth: **{len(bench)}** proteins "
          f"(TF={(bench.label=='TF').sum()}, TSP={(bench.label=='TSP').sum()}, "
          f"generic RBP={(bench.label=='RBP').sum()}, NEG={(bench.label=='NEG').sum()}).", "",
          "## Overall (binary)", "",
          "Positive class = RBP (TF|TSP|generic RBP) for general tools; TSP-only for "
          "the depolymerase/tailspike-specialised tools (DepoScope, SpikeHunter).", "",
          "| Tool | Task | TP | FP | TN | FN | Precision | Recall | Specificity | F1 | MCC |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows_overall:
        md.append("| {tool} | {task} | {TP} | {FP} | {TN} | {FN} | {precision:.3f} | "
                  "{recall:.3f} | {specificity:.3f} | {f1:.3f} | {mcc:.3f} |".format(**r))
    md += ["", "## Recall by ground-truth class", "",
           "| Tool | Recall TF | Recall TSP | Recall generic-RBP |", "|---|---|---|---|"]
    for r in rows_recall:
        md.append(f"| {r['tool']} | {r['recall_TF']:.3f} | {r['recall_TSP']:.3f} | "
                  f"{r['recall_RBP']:.3f} |")
    if has_tier and rows_tier:
        md += ["", "## By evidence tier", "",
               "| Tool | Tier | n | Precision | Recall | F1 | MCC |",
               "|---|---|---|---|---|---|---|"]
        for r in rows_tier:
            md.append(f"| {r['tool']} | T{r['tier']} | {r['n']} | {r['precision']:.3f} | "
                      f"{r['recall']:.3f} | {r['f1']:.3f} | {r['mcc']:.3f} |")
    if rbp_confusion is not None:
        md += ["", "## RBPdetect2 3-class confusion matrix", "",
               "Rows = true label, columns = predicted (0=NEG, 1=TF, 2=TSP).", "",
               "```", rbp_confusion.to_string(), "```"]
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")
    print(f"\nWrote {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"inphared", "experimental"}:
        sys.exit("usage: python score.py {inphared|experimental}")
    raise SystemExit(main(sys.argv[1]))
