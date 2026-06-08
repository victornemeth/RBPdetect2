from pathlib import Path

import pytest

from rbpdetect2.benchmark_data import load_labeled_sequences, load_saprot_sequences


def _write_processed_fastas(data_dir: Path) -> None:
    (data_dir / "tf.fasta").write_text(">tf1\nACD\n")
    (data_dir / "tsp.fasta").write_text(">tsp1\nEFG\n")
    (data_dir / "nonrbp.fasta").write_text(">neg1\nHIK\n")


def test_load_labeled_sequences_has_stable_class_order(tmp_path: Path) -> None:
    _write_processed_fastas(tmp_path)

    records = load_labeled_sequences(tmp_path)

    assert [(record["id"], record["label"]) for record in records] == [
        ("tf1", "TF"),
        ("tsp1", "TSP"),
        ("neg1", "nonRBP"),
    ]


def test_load_saprot_sequences_validates_interleaved_amino_acids(tmp_path: Path) -> None:
    _write_processed_fastas(tmp_path)
    records = load_labeled_sequences(tmp_path)
    path = tmp_path / "saprot.tsv"
    path.write_text("id\tsa_sequence\ntf1\tAaCbDc\ntsp1\tEdFeGg\nneg1\tHhIiKk\n")

    sequences = load_saprot_sequences(path, records)

    assert sequences["tf1"] == "AaCbDc"


def test_load_saprot_sequences_rejects_amino_acid_mismatch(tmp_path: Path) -> None:
    _write_processed_fastas(tmp_path)
    records = load_labeled_sequences(tmp_path)
    path = tmp_path / "saprot.tsv"
    path.write_text("id\tsa_sequence\ntf1\tXaCbDc\ntsp1\tEdFeGg\nneg1\tHhIiKk\n")

    with pytest.raises(ValueError, match="do not match FASTA"):
        load_saprot_sequences(path, records)

