from rbpdetect2.embedding_cli import _build_chunks


def test_build_chunks_uses_residue_coordinates() -> None:
    records = [{"id": "protein", "sequence": "ABCDEFG", "label": "TF"}]

    chunks = _build_chunks(
        records,
        max_residues=3,
        text_builder=lambda record, start, end: record["sequence"][start:end],
    )

    assert [chunk["text"] for chunk in chunks] == ["ABC", "DEF", "G"]
    assert [chunk["n_residues"] for chunk in chunks] == [3, 3, 1]


def test_build_chunks_supports_saprot_interleaved_text() -> None:
    records = [{"id": "protein", "sequence": "ABC", "label": "TF"}]
    saprot = "AaBbCc"

    chunks = _build_chunks(
        records,
        max_residues=2,
        text_builder=lambda _record, start, end: saprot[2 * start : 2 * end],
    )

    assert [chunk["text"] for chunk in chunks] == ["AaBb", "Cc"]

