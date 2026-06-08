"""Shared dataset utilities for embedding extraction and benchmarking."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


LABEL_FASTAS = {
    "TF": "tf.fasta",
    "TSP": "tsp.fasta",
    "nonRBP": "nonrbp.fasta",
}


def parse_fasta(path: Path) -> list[dict[str, str]]:
    """Read a FASTA file into records with stable first-token identifiers."""
    records: list[dict[str, str]] = []
    header: str | None = None
    sequence_parts: list[str] = []

    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append(_fasta_record(header, sequence_parts))
                header = line[1:]
                sequence_parts = []
            else:
                if header is None:
                    raise ValueError(f"Sequence found before FASTA header in {path}")
                sequence_parts.append(line)

    if header is not None:
        records.append(_fasta_record(header, sequence_parts))
    return records


def _fasta_record(header: str, sequence_parts: list[str]) -> dict[str, str]:
    sequence = "".join(sequence_parts).replace(" ", "").upper()
    if not sequence:
        raise ValueError(f"Empty sequence for FASTA record {header!r}")
    return {"id": header.split()[0], "header": header, "sequence": sequence}


def load_labeled_sequences(data_dir: Path) -> list[dict[str, str]]:
    """Load the processed TF, TSP, and nonRBP FASTAs in a stable order."""
    records: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    for label, filename in LABEL_FASTAS.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run scripts/combine_fastas.py first.")
        for record in parse_fasta(path):
            if record["id"] in seen_ids:
                raise ValueError(f"Duplicate sequence ID across processed FASTAs: {record['id']}")
            seen_ids.add(record["id"])
            records.append({**record, "label": label})

    return records


def dataset_sha256(records: list[dict[str, str]]) -> str:
    """Hash sequence IDs, labels, and sequences to identify an exact dataset."""
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["id"].encode())
        digest.update(b"\t")
        digest.update(record["label"].encode())
        digest.update(b"\t")
        digest.update(record["sequence"].encode())
        digest.update(b"\n")
    return digest.hexdigest()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()


def load_saprot_sequences(
    path: Path,
    records: list[dict[str, str]],
) -> dict[str, str]:
    """Load and validate SaProt AA+3Di sequences from a TSV file.

    Required columns are ``id`` and ``sa_sequence``. A SaProt sequence is an
    interleaved amino-acid and 3Di-token string, so it must have length 2L and
    its amino-acid characters must match the corresponding FASTA sequence.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing SaProt structure-aware sequence table: {path}")

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"id", "sa_sequence"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain tab-separated columns: id, sa_sequence")
        sa_sequences = {
            row["id"]: row["sa_sequence"].strip()
            for row in reader
            if row.get("id") and row.get("sa_sequence")
        }

    expected_ids = {record["id"] for record in records}
    missing = sorted(expected_ids - sa_sequences.keys())
    extra = sorted(sa_sequences.keys() - expected_ids)
    if missing:
        raise ValueError(f"SaProt table is missing {len(missing)} IDs; first missing ID: {missing[0]}")
    if extra:
        raise ValueError(f"SaProt table contains {len(extra)} unknown IDs; first extra ID: {extra[0]}")

    for record in records:
        sequence = record["sequence"]
        sa_sequence = sa_sequences[record["id"]]
        if len(sa_sequence) != 2 * len(sequence):
            raise ValueError(
                f"SaProt sequence length mismatch for {record['id']}: "
                f"expected {2 * len(sequence)}, got {len(sa_sequence)}"
            )
        if sa_sequence[::2].upper() != sequence:
            raise ValueError(f"SaProt amino-acid characters do not match FASTA for {record['id']}")

    return sa_sequences

