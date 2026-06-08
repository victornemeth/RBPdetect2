#!/usr/bin/env python3
"""Check benchmark sequences for overlap with the training set (leakage check).

For each benchmark set, runs MMseqs2 easy-search of the benchmark sequences
(query) against the combined training sequences (target = tf/tsp/nonrbp.fasta),
records the best hit per benchmark protein, and buckets by sequence identity.
High identity to a training protein means the benchmark result for that protein
may be inflated by memorisation rather than generalisation.

Also reports exact protein_id intersection between benchmark and training.

Writes per set:
    <set>/results/training_overlap.tsv      best hit per benchmark protein
    <set>/results/training_overlap_summary.md

Example:
    python scripts/check_benchmark_overlap.py
    python scripts/check_benchmark_overlap.py --sets inphared
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BENCH = ROOT / "benchmark"
TRAIN_FASTAS = [DATA / "tf.fasta", DATA / "tsp.fasta", DATA / "nonrbp.fasta"]
BUCKETS = [0.95, 0.70, 0.50, 0.30]  # report counts at >= each identity


def parse_fasta_ids(path: Path) -> set[str]:
    ids = set()
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                ids.add(line[1:].split()[0])
    return ids


def build_combined(paths: list[Path], dest: Path) -> None:
    with open(dest, "w") as out:
        for p in paths:
            with open(p) as f:
                shutil.copyfileobj(f, out)


def easy_search(query: Path, target: Path, tmp: Path) -> pd.DataFrame:
    """Return best hit per query: query, target, fident, qcov, tcov, evalue."""
    if not shutil.which("mmseqs"):
        raise RuntimeError("mmseqs not in PATH")
    out = tmp / "hits.m8"
    fmt = "query,target,fident,qcov,tcov,evalue,bits"
    subprocess.run([
        "mmseqs", "easy-search", str(query), str(target), str(out), str(tmp / "ms_tmp"),
        "-s", "7.5", "--max-seqs", "300",
        "-c", "0.0", "--cov-mode", "0",
        "--format-output", fmt, "-v", "1",
    ], check=True)
    cols = fmt.split(",")
    if out.stat().st_size == 0:
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(out, sep="\t", names=cols)
    # Best hit per query = highest bitscore (mmseqs already sorts, but be safe).
    df = df.sort_values("bits", ascending=False).drop_duplicates("query", keep="first")
    return df


def run_set(set_name: str, train_ids: set[str], combined: Path) -> None:
    base = BENCH / set_name
    fasta = base / "data" / "benchmark.fasta"
    out_dir = base / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_ids = parse_fasta_ids(fasta)
    exact = sorted(bench_ids & train_ids)

    with tempfile.TemporaryDirectory() as tmp:
        hits = easy_search(fasta, combined, Path(tmp))

    best = (
        pd.DataFrame({"protein_id": sorted(bench_ids)})
        .merge(hits.rename(columns={"query": "protein_id"}), on="protein_id", how="left")
    )
    best["fident"] = best["fident"].fillna(0.0)
    best["exact_id_match"] = best["protein_id"].isin(train_ids)
    best = best.sort_values("fident", ascending=False)
    best.to_csv(out_dir / "training_overlap.tsv", sep="\t", index=False)

    n = len(bench_ids)
    lines = [f"# Training-set overlap — {set_name}", "",
             f"Benchmark proteins: **{n}**", "",
             f"Exact protein_id matches with training set: **{len(exact)}**",
             ""]
    if exact:
        lines.append("Exact-id matches: " + ", ".join(exact[:50]) + (" ..." if len(exact) > 50 else ""))
        lines.append("")
    lines += ["## Best-hit identity to any training protein (MMseqs2, cov-mode 0)", "",
              "| identity ≥ | proteins | % of set |", "|---|---|---|"]
    for t in BUCKETS:
        k = int((best["fident"] >= t).sum())
        lines.append(f"| {int(t*100)}% | {k} | {100*k/n:.1f}% |")
    no_hit = int((best["fident"] == 0.0).sum())
    lines += [f"| no hit | {no_hit} | {100*no_hit/n:.1f}% |", "",
              f"Median best-hit identity: {best['fident'].median():.3f}; "
              f"mean: {best['fident'].mean():.3f}.", ""]
    (out_dir / "training_overlap_summary.md").write_text("\n".join(lines) + "\n")

    print(f"\n=== {set_name}: training-set overlap ===")
    print(f"benchmark proteins: {n}   exact protein_id matches: {len(exact)}")
    for t in BUCKETS:
        k = int((best["fident"] >= t).sum())
        print(f"  identity >= {int(t*100):>3}% : {k:>4}  ({100*k/n:4.1f}%)")
    print(f"  no hit         : {no_hit:>4}  ({100*no_hit/n:4.1f}%)")
    print(f"wrote {out_dir / 'training_overlap.tsv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=["inphared", "experimental"],
                    choices=["inphared", "experimental"])
    args = ap.parse_args()

    train_ids = set()
    for p in TRAIN_FASTAS:
        train_ids |= parse_fasta_ids(p)
    print(f"Training sequences: {len(train_ids)} across {len(TRAIN_FASTAS)} FASTAs")

    with tempfile.TemporaryDirectory() as tmp:
        combined = Path(tmp) / "train_all.fasta"
        build_combined(TRAIN_FASTAS, combined)
        for s in args.sets:
            run_set(s, train_ids, combined)


if __name__ == "__main__":
    main()
