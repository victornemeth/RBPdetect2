#!/usr/bin/env python3
"""Combine raw FASTA sources into tf.fasta, tsp.fasta, nonrbp.fasta in data/."""
import argparse
from pathlib import Path
from collections import OrderedDict


def parse_fasta(path: Path) -> OrderedDict:
    seqs = OrderedDict()
    header = None
    seq_parts = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    seqs[header] = "".join(seq_parts)
                header = line[1:]
                seq_parts = []
            elif line:
                seq_parts.append(line)
    if header is not None:
        seqs[header] = "".join(seq_parts)
    return seqs


def write_fasta(seqs: OrderedDict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for header, seq in seqs.items():
            f.write(f">{header}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")


# Sequences confirmed non-RBP (misannotated) — excluded from all processed outputs.
# CAB63592.1: DNA-directed RNA polymerase, found in tf_annotated.fasta
# CAJ29347.1: DNA-directed RNA polymerase, found in tsp_annotated.fasta
# MW358930_CDS_0001: mislabeled nonRBP, manual PCA review flagged it as a TSP.
EXCLUDED_IDS: set[str] = {
    "CAB63592.1",
    "CAJ29347.1",
    "MW358930_CDS_0001",
}


def combine(sources: list[Path], label: str) -> OrderedDict:
    combined = OrderedDict()
    seen_ids = set()
    for src in sources:
        seqs = parse_fasta(src)
        for header, seq in seqs.items():
            seq_id = header.split()[0]
            if seq_id in EXCLUDED_IDS:
                print(f"  EXCLUDE {seq_id} (from {src.name})")
                continue
            if seq_id in seen_ids:
                print(f"  SKIP duplicate id: {seq_id} (from {src.name})")
                continue
            seen_ids.add(seq_id)
            # Strip translation stop codons (present in nonRBP source FASTAs);
            # structure predictors drop them, so they must not enter the seqs.
            combined[header] = seq.replace("*", "")
    print(f"{label}: {len(combined)} sequences")
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw", type=Path)
    parser.add_argument("--out-dir", default="data", type=Path)
    args = parser.parse_args()

    raw = args.raw_dir
    out = args.out_dir

    tf = combine(
        [raw / "tf_annotated.fasta", raw / "TF_examples.fasta", raw / "TF.fasta"],
        "TF",
    )
    tsp = combine(
        [
            raw / "tsp_annotated.fasta",
            raw / "TSP_DepoCat.fasta",
            raw / "deposcope_nozzle_filtered.fasta",
        ],
        "TSP",
    )
    nonrbp = combine(
        [raw / "nonRBPs.fasta"],
        "nonRBP",
    )

    write_fasta(tf, out / "tf.fasta")
    write_fasta(tsp, out / "tsp.fasta")
    write_fasta(nonrbp, out / "nonrbp.fasta")
    print(f"Written to {out}/")


if __name__ == "__main__":
    main()
