#!/usr/bin/env python3
"""Filter manually-annotated DepoScope positives.

Drops sequences annotated "Wrong" in annotations_DepoScope.tsv, keeping
"Correct" and "Unsure". Writes the survivors to data/raw so combine_fastas.py
folds them into tsp.fasta.

Protein IDs differ in case between the FASTA (e.g. rcsb_pdb_2YW0) and the
annotation table (RCSB_PDB_2YW0), so matching is case-insensitive.
"""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fasta", type=Path, default=Path("data/deposcope/deposcope_positives.fasta")
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/deposcope/annotations_DepoScope.tsv"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/raw/deposcope_nozzle_filtered.fasta")
    )
    args = parser.parse_args()

    # Collect IDs annotated "Wrong" (case-insensitive key).
    wrong = set()
    with open(args.annotations) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["Annotation"].strip().lower() == "wrong":
                wrong.add(row["Protein ID"].strip().lower())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    keep = True
    with open(args.fasta) as fin, open(args.out, "w") as fout:
        for line in fin:
            if line.startswith(">"):
                seq_id = line[1:].split()[0]
                keep = seq_id.lower() not in wrong
                if keep:
                    kept += 1
                else:
                    dropped += 1
            if keep:
                fout.write(line)

    print(f"Wrong annotations: {len(wrong)}")
    print(f"Kept: {kept}  Dropped: {dropped}")
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
