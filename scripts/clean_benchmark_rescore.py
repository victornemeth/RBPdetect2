#!/usr/bin/env python3
"""Re-score every tool on benchmark subsets with our-training overlap removed.

For each benchmark set and each dedup level, drop the benchmark proteins that
overlap our training set, then re-run benchmark/score.py on the remainder so
EVERY tool is measured on the same leakage-reduced subset.

Dedup levels (overlap = shares similarity with any training sequence):
    id100   exact sequence match to a training protein
    id90    >=90% identity with >=80% of the benchmark protein aligned
    id70    >=70% identity, same coverage gate
    id50    >=50% identity, same coverage gate

IMPORTANT asymmetry: overlap is removed only against *our* training set. The
baseline tools (Phold, Pharokka, v4, DepoScope, ...) were trained on their own
data we cannot dedup against, so this neutralises OUR leakage advantage, not
theirs. It is the fair-as-possible comparison given available data.

Filtered sets are written under benchmark/clean/<set>_<level>/ and scored with
the existing score.py; a consolidated table lands in
benchmark/RESULTS_clean_rescore.md.

Example:
    python scripts/clean_benchmark_rescore.py
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmark"))
import score  # noqa: E402  (benchmark/score.py)

DATA = ROOT / "data"
BENCH = ROOT / "benchmark"
CLEAN = BENCH / "clean"
TRAIN_FASTAS = [DATA / "tf.fasta", DATA / "tsp.fasta", DATA / "nonrbp.fasta"]
QCOV_GATE = 0.8                      # benchmark protein must be >=80% aligned
LEVELS = [("id100", None), ("id90", 0.90), ("id70", 0.70), ("id50", 0.50)]
SETS = ["inphared", "experimental"]


def parse_fasta(path: Path) -> dict[str, str]:
    seqs, header, parts = {}, None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    seqs[header.split()[0]] = "".join(parts)
                header, parts = line[1:], []
            elif line:
                parts.append(line)
    if header is not None:
        seqs[header.split()[0]] = "".join(parts)
    return seqs


def easy_search(query: Path, target: Path, tmp: Path) -> pd.DataFrame:
    if not shutil.which("mmseqs"):
        raise RuntimeError("mmseqs not in PATH")
    out = tmp / "hits.m8"
    fmt = "query,target,fident,qcov,tcov,bits"
    subprocess.run([
        "mmseqs", "easy-search", str(query), str(target), str(out), str(tmp / "ms_tmp"),
        "-s", "7.5", "--max-seqs", "300", "-c", "0.0", "--cov-mode", "0",
        "--format-output", fmt, "-v", "1",
    ], check=True)
    cols = fmt.split(",")
    if out.stat().st_size == 0:
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(out, sep="\t", names=cols)
    return df.sort_values("bits", ascending=False).drop_duplicates("query", keep="first")


def overlap_ids(set_name: str, train_seqs: dict[str, str], combined: Path) -> dict[str, set[str]]:
    """Return {level: set of benchmark protein_ids to EXCLUDE}."""
    bench_seqs = parse_fasta(BENCH / set_name / "data" / "benchmark.fasta")
    train_seq_set = set(train_seqs.values())

    # id100: exact sequence string match
    exact = {pid for pid, s in bench_seqs.items() if s in train_seq_set}

    with tempfile.TemporaryDirectory() as tmp:
        hits = easy_search(BENCH / set_name / "data" / "benchmark.fasta", combined, Path(tmp))
    hits = hits[hits["qcov"] >= QCOV_GATE]
    best = hits.set_index("query")["fident"]

    result = {"id100": exact}
    for level, thr in LEVELS:
        if thr is None:
            continue
        ids = set(best[best >= thr].index)
        result[level] = ids | exact   # exact dups are a subset of every identity cut
    return result


def make_clean_set(set_name: str, level: str, exclude: set[str]) -> str:
    """Write benchmark/clean/<set>_<level>/ with filtered tsv + symlinked predictions."""
    src = BENCH / set_name
    rel = f"clean/{set_name}_{level}"
    dest = BENCH / rel
    (dest / "data").mkdir(parents=True, exist_ok=True)

    bench = pd.read_csv(src / "data" / "benchmark.tsv", sep="\t", dtype=str, keep_default_na=False)
    kept = bench[~bench["protein_id"].isin(exclude)]
    kept.to_csv(dest / "data" / "benchmark.tsv", sep="\t", index=False)

    # Reuse the existing predictions unchanged (score.py left-joins onto the
    # filtered ground truth, so extra prediction rows are ignored).
    pred_link = dest / "predictions"
    if pred_link.is_symlink() or pred_link.exists():
        if pred_link.is_symlink():
            pred_link.unlink()
        else:
            shutil.rmtree(pred_link)
    pred_link.symlink_to(src / "predictions", target_is_directory=True)
    return rel


def main() -> None:
    train_seqs: dict[str, str] = {}
    for p in TRAIN_FASTAS:
        train_seqs.update(parse_fasta(p))
    print(f"Training sequences: {len(train_seqs)}")

    summary_rows = []   # (set, level, n_kept, n_removed, n_pos)
    metric_rows = []    # one row per (set, level, tool)

    with tempfile.TemporaryDirectory() as tmp:
        combined = Path(tmp) / "train_all.fasta"
        with open(combined, "w") as out:
            for p in TRAIN_FASTAS:
                with open(p) as f:
                    shutil.copyfileobj(f, out)

        for set_name in SETS:
            excl_by_level = overlap_ids(set_name, train_seqs, combined)
            full = pd.read_csv(BENCH / set_name / "data" / "benchmark.tsv", sep="\t",
                               dtype=str, keep_default_na=False)
            n_full = len(full)

            for level in ["all", "id100", "id90", "id70", "id50"]:
                exclude = set() if level == "all" else excl_by_level[level]
                rel = make_clean_set(set_name, "full" if level == "all" else level, exclude)
                print(f"\n########## {set_name} / {level} "
                      f"(removed {len(exclude & set(full.protein_id))}/{n_full}) ##########")
                score.main(rel)

                kept = full[~full["protein_id"].isin(exclude)]
                n_pos = int((kept["label"] != "NEG").sum())
                summary_rows.append({"set": set_name, "level": level,
                                     "n_kept": len(kept), "n_removed": n_full - len(kept),
                                     "n_pos": n_pos})

                mo = pd.read_csv(BENCH / rel / "results" / "metrics_overall.tsv", sep="\t")
                for _, r in mo.iterrows():
                    metric_rows.append({"set": set_name, "level": level, "tool": r["tool"],
                                        "task": r["task"], "f1": r["f1"], "recall": r["recall"],
                                        "precision": r["precision"], "mcc": r["mcc"]})

    summary = pd.DataFrame(summary_rows)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(CLEAN / "metrics_long.tsv", sep="\t", index=False)

    write_report(summary, metrics)
    print(f"\nWrote {BENCH / 'RESULTS_clean_rescore.md'}")


def write_report(summary: pd.DataFrame, metrics: pd.DataFrame) -> None:
    order = ["all", "id100", "id90", "id70", "id50"]
    md = ["# Clean-subset re-score — overlap with our training removed", "",
          "Every tool re-scored on benchmark subsets after removing proteins that",
          "overlap **our** training set at each dedup level. Overlap = exact match",
          f"(`id100`) or >=X% identity with >={int(QCOV_GATE*100)}% of the benchmark",
          "protein aligned (`id90/id70/id50`). Metric = RBP-binary F1",
          "(TSP-only F1 for DepoScope/SpikeHunter).", "",
          "⚠️ Dedup is vs OUR training only; baselines' training data is unavailable,",
          "so their leakage is not removed. This removes our advantage, not theirs.", ""]

    for set_name in SETS:
        s = summary[summary.set == set_name].set_index("level")
        md += [f"## {set_name}", "",
               "Subset sizes (proteins kept / removed / positives):", "",
               "| level | kept | removed | positives |", "|---|---|---|---|"]
        for lv in order:
            if lv in s.index:
                row = s.loc[lv]
                md.append(f"| {lv} | {row.n_kept} | {row.n_removed} | {row.n_pos} |")
        md += ["", "F1 by tool across dedup levels:", "",
               "| Tool | " + " | ".join(order) + " |",
               "|---|" + "|".join(["---"] * len(order)) + "|"]
        msub = metrics[metrics.set == set_name]
        for tool in msub["tool"].drop_duplicates():
            cells = []
            for lv in order:
                v = msub[(msub.level == lv) & (msub.tool == tool)]["f1"]
                cells.append(f"{v.iloc[0]:.3f}" if len(v) else "—")
            md.append(f"| {tool} | " + " | ".join(cells) + " |")
        md.append("")

    (BENCH / "RESULTS_clean_rescore.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
