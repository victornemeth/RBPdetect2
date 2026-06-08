#!/usr/bin/env python3
"""Per-class (TF / TSP) and binary (RBP) metric tables, per dedup level.

Builds, for every benchmark set and every overlap-dedup level, three things per
tool: binary RBP detection metrics, and one-vs-rest TF and TSP metrics
(precision / recall / F1 / MCC). Output: benchmark/RESULTS_metric_tables.md.

Subtype derivation (each tool's prediction -> NEG / TF / TSP call):
  rbpdetect2 (new/prev) : pred 0=NEG, 1=TF, 2=TSP      (native 3-class)
  blastp                : top_label none/TF/TSP         (native 3-class)
  phold                 : pred==0 -> NEG; product contains spike|depolymer
                          -> TSP, else -> TF            (product-derived)
  pharokka, v4          : positive -> TF (cannot subtype; TSP never predicted,
                          so TSP column is 0 by construction)
  phanns                : Tail-fiber class -> TF (no TSP class -> TSP=0)
  deposcope, spikehunter: TSP-only specialists -> TSP; TF not applicable ("—")

⚠️ Pharokka / v4 / PhANNs cannot call TSP, so their TSP row is 0 by construction
(not a measured failure). DepoScope / SpikeHunter are TSP-only, so their TF row
is omitted. Dedup is vs OUR training only (baselines' training unavailable).

Reuses the clean subsets built by scripts/clean_benchmark_rescore.py
(benchmark/clean/<set>_<level>/data/benchmark.tsv).

Example:
    python scripts/per_class_tables.py
"""

import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmark"

SETS = ["inphared", "experimental"]
LEVELS = ["all", "id100", "id90", "id70", "id50"]

# (filename, display, kind) — kind: "general" (TF+TSP), "tsp" (TSP-only)
TOOLS = [
    ("rbpdetect_new.tsv", "RBPdetect2 (new, this work)",          "general"),
    ("rbpdetect2.tsv",    "RBPdetect2 (previous)",                "general"),
    ("rbpdetect_v4.tsv",  "PhageRBPdetect v4 (Boeckaerts 2024)",  "general"),
    ("phold.tsv",         "Phold (Bouras 2024)",                  "general"),
    ("pharokka.tsv",      "Pharokka (Bouras 2023)",               "general"),
    ("phanns.tsv",        "PhANNs (Cantu 2020)",                  "general"),
    ("blastp.tsv",        "BLASTp baseline",                      "general"),
    ("deposcope.tsv",     "DepoScope (Concha-Eloko 2024)",        "tsp"),
    ("spikehunter.tsv",   "SpikeHunter (Yang 2024)",              "tsp"),
]


def three_way_call(fname: str, pred: pd.DataFrame) -> pd.Series:
    """Map a tool's prediction TSV to a NEG/TF/TSP call per protein_id."""
    p = pred.set_index("protein_id")
    if fname in ("rbpdetect_new.tsv", "rbpdetect2.tsv"):
        m = {0: "NEG", 1: "TF", 2: "TSP"}
        return p["pred"].astype(int).map(m)
    if fname == "blastp.tsv":
        return p["top_label"].map(lambda x: x if x in ("TF", "TSP") else "NEG")
    if fname == "phold.tsv":
        def call(row):
            if int(row["pred"]) == 0:
                return "NEG"
            prod = str(row["product"]).lower()
            return "TSP" if ("spike" in prod or "depolymer" in prod) else "TF"
        return p.apply(call, axis=1)
    if fname in ("rbpdetect_v4.tsv", "pharokka.tsv", "phanns.tsv"):
        return p["pred"].astype(int).map(lambda v: "TF" if v == 1 else "NEG")
    if fname == "deposcope.tsv":
        return p["pred"].astype(int).map(lambda v: "TSP" if v == 1 else "NEG")
    if fname == "spikehunter.tsv":
        return p["Predicted_label"].map(lambda x: "TSP" if float(x) != 0 else "NEG")
    raise ValueError(fname)


def metrics(y_true, y_pred) -> dict:
    has_both = len(set(y_true)) > 1 and len(set(y_pred)) > 1
    return {
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if has_both else 0.0,
    }


def fmt(v) -> str:
    return "—" if v is None else f"{v:.3f}"


def cells(m: dict | None) -> str:
    if m is None:
        return " — | — | — | — "
    return f" {m['prec']:.3f} | {m['rec']:.3f} | {m['f1']:.3f} | {m['mcc']:.3f} "


def load_pred(set_name: str, fname: str) -> pd.DataFrame | None:
    fpath = BENCH / set_name / "predictions" / fname
    if not fpath.exists():
        return None
    return pd.read_csv(fpath, sep="\t", dtype={"protein_id": str})


def subset_ids(set_name: str, level: str) -> pd.DataFrame:
    rel = f"{set_name}_full" if level == "all" else f"{set_name}_{level}"
    path = BENCH / "clean" / rel / "data" / "benchmark.tsv"
    if not path.exists():  # "all" fallback to original
        path = BENCH / set_name / "data" / "benchmark.tsv"
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def build() -> None:
    md = ["# Per-class & binary metric tables (by dedup level)", "",
          "Precision / Recall / F1 / MCC. Binary = RBP detection (positive = "
          "TF|TSP|generic RBP); TF and TSP = one-vs-rest. Subtype derivation and "
          "caveats: see header of scripts/per_class_tables.py.", "",
          "Dedup levels remove benchmark proteins overlapping **our** training: "
          "`id100` exact match, `id90/70/50` ≥X% identity over ≥80% of the protein.",
          ""]

    for set_name in SETS:
        md += [f"# {set_name}", ""]
        # precompute each tool's 3-way call once (over all proteins)
        calls = {}
        for fname, _, _ in TOOLS:
            pred = load_pred(set_name, fname)
            calls[fname] = None if pred is None else three_way_call(fname, pred)

        for level in LEVELS:
            bench = subset_ids(set_name, level)
            ids = bench["protein_id"].tolist()
            label = bench.set_index("protein_id")["label"]
            n = len(bench)
            n_tf = int((label == "TF").sum())
            n_tsp = int((label == "TSP").sum())
            n_pos = int((label != "NEG").sum())

            md += [f"## {set_name} — {level}  "
                   f"(n={n}, TF={n_tf}, TSP={n_tsp}, RBP-pos={n_pos})", ""]

            # ---- binary RBP table ----
            md += ["### Binary (RBP detection)", "",
                   "| Tool | Prec | Rec | F1 | MCC |", "|---|---|---|---|---|"]
            for fname, disp, kind in TOOLS:
                call = calls[fname]
                if call is None:
                    continue
                c = call.reindex(ids).fillna("NEG")
                if kind == "tsp":
                    y_true = (label.reindex(ids) == "TSP").astype(int).values
                    y_pred = (c == "TSP").astype(int).values
                    tag = " (TSP-only)"
                else:
                    y_true = (label.reindex(ids) != "NEG").astype(int).values
                    y_pred = (c != "NEG").astype(int).values
                    tag = ""
                m = metrics(y_true, y_pred)
                md.append(f"| {disp}{tag} |{cells(m)}|")
            md.append("")

            # ---- per-class TF / TSP table ----
            md += ["### Per-class (TF / TSP, one-vs-rest)", "",
                   "| Tool | TF Prec | TF Rec | TF F1 | TF MCC "
                   "| TSP Prec | TSP Rec | TSP F1 | TSP MCC |",
                   "|---|---|---|---|---|---|---|---|---|"]
            yt_tf = (label.reindex(ids) == "TF").astype(int).values
            yt_tsp = (label.reindex(ids) == "TSP").astype(int).values
            for fname, disp, kind in TOOLS:
                call = calls[fname]
                if call is None:
                    continue
                c = call.reindex(ids).fillna("NEG")
                tf_m = None if kind == "tsp" else metrics(yt_tf, (c == "TF").astype(int).values)
                tsp_m = metrics(yt_tsp, (c == "TSP").astype(int).values)
                md.append(f"| {disp} |{cells(tf_m)}|{cells(tsp_m)}|")
            md.append("")

    out = BENCH / "RESULTS_metric_tables.md"
    out.write_text("\n".join(md) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    build()
