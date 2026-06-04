#!/usr/bin/env python3
"""
Sequence diversity analysis using MMseqs2.

Per-group clustering at multiple identity thresholds + cross-group contamination check.
Outputs TSV summary tables and prints a brief report.
"""
import argparse
import csv
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path


THRESHOLDS = [0.3, 0.5, 0.7, 0.9]
GROUPS = ["tf", "tsp", "nonrbp"]


def run(cmd: list[str], **kwargs):
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, **kwargs)
    return result


def mmseqs_easy_cluster(fasta: Path, out_prefix: Path, tmp: Path, min_id: float, cov: float = 0.8):
    run([
        "mmseqs", "easy-cluster",
        str(fasta), str(out_prefix), str(tmp),
        "--min-seq-id", str(min_id),
        "-c", str(cov),
        "--cov-mode", "0",
        "--cluster-mode", "2",
        "-v", "0",
    ])


def parse_cluster_tsv(tsv_path: Path) -> dict[str, str]:
    """Return {member_id: rep_id}."""
    member_to_rep = {}
    with open(tsv_path) as f:
        for line in f:
            rep, member = line.strip().split("\t")
            member_to_rep[member] = rep
    return member_to_rep


def cluster_stats(member_to_rep: dict) -> dict:
    clusters = defaultdict(list)
    for member, rep in member_to_rep.items():
        clusters[rep].append(member)
    sizes = [len(v) for v in clusters.values()]
    return {
        "n_seqs": len(member_to_rep),
        "n_clusters": len(clusters),
        "largest_cluster": max(sizes),
        "singletons": sum(1 for s in sizes if s == 1),
    }


def mmseqs_easy_search(query: Path, target: Path, out: Path, tmp: Path, min_id: float, cov: float = 0.5):
    run([
        "mmseqs", "easy-search",
        str(query), str(target), str(out), str(tmp),
        "--min-seq-id", str(min_id),
        "-c", str(cov),
        "--cov-mode", "0",
        "--format-output", "query,target,pident,qcov,tcov,evalue",
        "-v", "0",
    ])


def parse_hits(tsv_path: Path) -> list[dict]:
    hits = []
    if not tsv_path.exists():
        return hits
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            hits.append({
                "query": parts[0],
                "target": parts[1],
                "pident": float(parts[2]),
                "qcov": float(parts[3]),
                "tcov": float(parts[4]),
                "evalue": float(parts[5]),
            })
    return hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--out-dir", default="data/diversity", type=Path)
    parser.add_argument(
        "--contamination-id",
        default=0.5,
        type=float,
        help="Min identity for cross-group contamination check",
    )
    args = parser.parse_args()

    if not shutil.which("mmseqs"):
        raise SystemExit("mmseqs not found in PATH. Install with: conda install -c bioconda mmseqs2")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    fastas = {g: args.data_dir / f"{g}.fasta" for g in GROUPS}
    for g, p in fastas.items():
        if not p.exists():
            raise SystemExit(f"Missing {p}. Run scripts/combine_fastas.py first.")

    # ── 1. Per-group clustering ─────────────────────────────────────────────
    print("\n=== Per-group clustering ===")
    cluster_summary_rows = []

    with tempfile.TemporaryDirectory() as tmp_root:
        tmp = Path(tmp_root)

        for group in GROUPS:
            fasta = fastas[group]
            for thresh in THRESHOLDS:
                prefix = args.out_dir / f"{group}_clust_{int(thresh*100)}"
                cluster_tmp = tmp / f"{group}_{int(thresh*100)}"
                cluster_tmp.mkdir()
                mmseqs_easy_cluster(fasta, prefix, cluster_tmp, min_id=thresh)
                tsv = Path(str(prefix) + "_cluster.tsv")
                stats = cluster_stats(parse_cluster_tsv(tsv))
                row = {"group": group, "identity": thresh, **stats}
                cluster_summary_rows.append(row)
                print(
                    f"  {group:6s} @{int(thresh*100):3d}%: "
                    f"{stats['n_clusters']:4d} clusters / {stats['n_seqs']:4d} seqs "
                    f"(largest={stats['largest_cluster']}, singletons={stats['singletons']})"
                )

        # ── 2. Cross-group contamination ────────────────────────────────────
        print(f"\n=== Cross-group contamination (id≥{args.contamination_id}) ===")
        cross_rows = []
        pairs = [("tf", "tsp"), ("tf", "nonrbp"), ("tsp", "nonrbp")]

        for q_name, t_name in pairs:
            out_tsv = args.out_dir / f"cross_{q_name}_vs_{t_name}.tsv"
            cross_tmp = tmp / f"cross_{q_name}_{t_name}"
            cross_tmp.mkdir()
            mmseqs_easy_search(
                fastas[q_name], fastas[t_name], out_tsv, cross_tmp,
                min_id=args.contamination_id,
            )
            hits = parse_hits(out_tsv)
            unique_queries = len({h["query"] for h in hits})
            unique_targets = len({h["target"] for h in hits})
            print(
                f"  {q_name} vs {t_name}: {len(hits)} hits, "
                f"{unique_queries} {q_name} seqs with hit, "
                f"{unique_targets} {t_name} seqs matched"
            )
            cross_rows.append({
                "query_group": q_name,
                "target_group": t_name,
                "n_hits": len(hits),
                "unique_query_seqs": unique_queries,
                "unique_target_seqs": unique_targets,
            })
            if hits:
                flagged = args.out_dir / f"flagged_{q_name}_vs_{t_name}.tsv"
                with open(flagged, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=hits[0].keys(), delimiter="\t")
                    writer.writeheader()
                    writer.writerows(hits)
                print(f"    → flagged sequences written to {flagged}")

    # ── 3. Write summary tables ──────────────────────────────────────────────
    clust_out = args.out_dir / "cluster_summary.tsv"
    with open(clust_out, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["group", "identity", "n_seqs", "n_clusters", "largest_cluster", "singletons"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(cluster_summary_rows)

    cross_out = args.out_dir / "contamination_summary.tsv"
    with open(cross_out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["query_group", "target_group", "n_hits", "unique_query_seqs", "unique_target_seqs"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(cross_rows)

    print(f"\nSummaries written to {args.out_dir}/")


if __name__ == "__main__":
    main()
