#!/usr/bin/env python3
"""Report which sequences in tf/tsp/nonrbp FASTA lack a predicted structure.

Structures live as ``{protein_id}.pdb`` under data/structures/{tf,deposcope,
positives,nonrbps}. Matching is case-insensitive.

ID resolution: TSP_DepoCat headers are pipe-delimited and carry the accession
as ``Protein_ID=<acc>`` rather than as the first token, so that field is
preferred when present. Accessions of ``n_a`` / ``-----`` are unusable (no
accession exists) and such sequences are reported as missing.

Writes the still-missing sequences per group to
data/structures/missing/{group}_missing.fasta for structure prediction.
"""
import argparse
import re
from pathlib import Path

UNUSABLE = re.compile(r"^(n_a|-+|)$", re.IGNORECASE)
PROTID = re.compile(r"Protein_ID=([^|]*)")


def resolve_id(header: str) -> str:
    m = PROTID.search(header)
    if m:
        acc = m.group(1).strip()
        if not UNUSABLE.match(acc):
            return acc
        return ""  # unusable accession -> no structure can match
    return header.split()[0]


def parse_fasta(path: Path):
    header, parts = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(parts)
                header, parts = line[1:], []
            elif line:
                parts.append(line)
    if header is not None:
        yield header, "".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--struct-dir", type=Path, default=Path("data/structures"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/structures/missing"))
    args = ap.parse_args()

    struct = {}
    for folder in ["tf", "deposcope", "positives", "nonrbps"]:
        for p in (args.struct_dir / folder).glob("*.pdb"):
            struct.setdefault(p.stem.lower(), set()).add(folder)
    print(f"indexed {len(struct)} structure ids")

    groups = {
        "tf": Path("data/tf.fasta"),
        "tsp": Path("data/tsp.fasta"),
        "nonrbp": Path("data/nonrbp.fasta"),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name, fa in groups.items():
        records = list(parse_fasta(fa))
        missing = []
        for header, seq in records:
            sid = resolve_id(header)
            if not sid or sid.lower() not in struct:
                missing.append((header, seq))
        print(f"\n{name}: {len(records)} seqs | have {len(records)-len(missing)} | MISSING {len(missing)}")
        out = args.out_dir / f"{name}_missing.fasta"
        with open(out, "w") as f:
            for header, seq in missing:
                f.write(f">{header}\n")
                for i in range(0, len(seq), 60):
                    f.write(seq[i:i+60] + "\n")
        print(f"   -> {out}")


if __name__ == "__main__":
    main()
