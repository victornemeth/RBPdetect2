#!/usr/bin/env python3
"""Build the SaProt AA+3Di sequence table from predicted structures.

For every labeled record that has a predicted structure, extract the 3Di
structural alphabet with foldseek and interleave it with the amino-acid
sequence to form SaProt input (``A`` + ``d`` -> ``Ad`` ...). Records without a
structure, or whose structure residues do not match the FASTA sequence, are
skipped and reported. The resulting table (data/saprot_sequences.tsv) is
therefore "available structures only".

Structure lookup mirrors scripts/check_structures.py: TSP_DepoCat headers carry
the accession as ``Protein_ID=<acc>`` rather than as the first token.
"""
import argparse
import re
import subprocess
import tempfile
from pathlib import Path

from rbpdetect2.benchmark_data import load_labeled_sequences

UNUSABLE = re.compile(r"^(n_a|-+|)$", re.IGNORECASE)
PROTID = re.compile(r"Protein_ID=([^|]*)")
FOLDSEEK = Path.home() / "miniconda3/envs/pholdENV/bin/foldseek"
# Ambiguous / non-standard codes that foldseek renders as X.
AMBIGUOUS = set("XBZJUO")
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def aa_compatible(struct_aa: str, seq: str) -> bool:
    """True if foldseek's per-residue AA seq matches the FASTA seq, allowing
    ambiguous FASTA codes (X/B/Z/J/U/O) or an X from foldseek as wildcards."""
    if len(struct_aa) != len(seq):
        return False
    for a, b in zip(struct_aa.upper(), seq):
        if a == b or a == "X" or b in AMBIGUOUS:
            continue
        return False
    return True


def resolve_id(header: str) -> str:
    m = PROTID.search(header)
    if m:
        acc = m.group(1).strip()
        return "" if UNUSABLE.match(acc) else acc
    return header.split()[0]


def index_structures(struct_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for folder in ["tf", "deposcope", "positives", "nonrbps"]:
        for p in (struct_dir / folder).glob("*.pdb"):
            index.setdefault(p.stem.lower(), p)
    return index


def foldseek_3di(pdb: Path) -> tuple[str, str] | None:
    """Return (aa_seq, 3di_seq) for the first chain, or None on failure."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "desc"
        proc = subprocess.run(
            [str(FOLDSEEK), "structureto3didescriptor", "-v", "0",
             "--threads", "1", "--chain-name-mode", "1", str(pdb), str(out)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0 or not out.exists():
            return None
        line = out.read_text().splitlines()
        if not line:
            return None
        cols = line[0].split("\t")
        if len(cols) < 3:
            return None
        return cols[1], cols[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--struct-dir", type=Path, default=Path("data/structures"))
    ap.add_argument("--out", type=Path, default=Path("data/saprot_sequences.tsv"))
    args = ap.parse_args()

    records = load_labeled_sequences(args.data_dir)
    index = index_structures(args.struct_dir)
    print(f"{len(records)} records | {len(index)} structures indexed")

    cache: dict[Path, tuple[str, str] | None] = {}
    rows: list[tuple[str, str]] = []
    no_struct = mismatch = fs_fail = nonstd = 0

    for rec in records:
        # SaProt's tokenizer emits exactly one token per standard residue;
        # non-standard codes (X/B/Z/J/U/O) break that 1:1 mapping, so skip them.
        if set(rec["sequence"]) - STANDARD_AA:
            nonstd += 1
            continue
        rid = resolve_id(rec["header"])
        pdb = index.get(rid.lower()) if rid else None
        if pdb is None:
            no_struct += 1
            continue
        if pdb not in cache:
            cache[pdb] = foldseek_3di(pdb)
        res = cache[pdb]
        if res is None:
            fs_fail += 1
            continue
        aa, threedi = res
        seq = rec["sequence"]
        if len(threedi) != len(seq) or not aa_compatible(aa, seq):
            mismatch += 1
            continue
        sa = "".join(a + d.lower() for a, d in zip(seq, threedi))
        rows.append((rec["id"], sa))

    with open(args.out, "w") as f:
        f.write("id\tsa_sequence\n")
        for rid, sa in rows:
            f.write(f"{rid}\t{sa}\n")

    print(f"written: {len(rows)} | no_structure: {no_struct} | "
          f"nonstd_aa_skipped: {nonstd} | aa_mismatch: {mismatch} | foldseek_fail: {fs_fail}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
