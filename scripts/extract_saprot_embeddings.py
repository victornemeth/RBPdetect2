#!/usr/bin/env python3
"""Extract frozen SaProt-650M structure-aware embeddings for the benchmark dataset."""

import argparse
from pathlib import Path

from rbpdetect2.benchmark_data import load_labeled_sequences, load_saprot_sequences
from rbpdetect2.embedding_cli import add_common_arguments, run_hf_extractor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(
        parser,
        default_model_id="westlake-repl/SaProt_650M_AF2",
        default_output_dir="benchmarks/embeddings/saprot-650m",
    )
    parser.add_argument(
        "--sa-sequences",
        type=Path,
        default=Path("data/saprot_sequences.tsv"),
        help="TSV with columns id and sa_sequence containing interleaved AA+3Di strings",
    )
    args = parser.parse_args()

    records = load_labeled_sequences(args.data_dir)
    # SaProt covers only sequences with a usable structure; restrict the record
    # set to those present in the AA+3Di table ("available structures only").
    table_ids = _table_ids(args.sa_sequences)
    available = [record for record in records if record["id"] in table_ids]
    skipped = len(records) - len(available)
    print(f"SaProt: {len(available)} of {len(records)} records have structures ({skipped} skipped)")
    sa_sequences = load_saprot_sequences(args.sa_sequences, available)

    def text_builder(record: dict[str, str], start: int, end: int) -> str:
        return sa_sequences[record["id"]][2 * start : 2 * end]

    run_hf_extractor(
        args,
        model_key="saprot-650m",
        text_builder=text_builder,
        input_kind="saprot_interleaved_amino_acid_and_3di_sequence",
        records=available,
    )


def _table_ids(path: Path) -> set[str]:
    import csv

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {row["id"] for row in reader if row.get("id")}


if __name__ == "__main__":
    main()

