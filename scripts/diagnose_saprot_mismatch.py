#!/usr/bin/env python3
"""Diagnose why foldseek structure AA sequences disagree with the FASTA seqs."""
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from rbpdetect2.benchmark_data import load_labeled_sequences

UNUSABLE = re.compile(r"^(n_a|-+|)$", re.IGNORECASE)
PROTID = re.compile(r"Protein_ID=([^|]*)")
FS = Path.home() / "miniconda3/envs/pholdENV/bin/foldseek"


def rid(h):
    m = PROTID.search(h)
    if m:
        a = m.group(1).strip()
        return "" if UNUSABLE.match(a) else a
    return h.split()[0]


def index():
    d = {}
    for f in ["tf", "deposcope", "positives", "nonrbps"]:
        for p in Path("data/structures", f).glob("*.pdb"):
            d.setdefault(p.stem.lower(), (p, f))
    return d


def foldseek_aa(pdb):
    with tempfile.TemporaryDirectory() as t:
        o = Path(t) / "o"
        r = subprocess.run(
            [str(FS), "structureto3didescriptor", "-v", "0", "--threads", "1",
             "--chain-name-mode", "1", str(pdb), str(o)],
            capture_output=True,
        )
        if r.returncode != 0 or not o.exists():
            return None
        lines = o.read_text().splitlines()
        return lines[0].split("\t")[1] if lines else None


def main():
    recs = load_labeled_sequences(Path("data"))
    I = index()

    cat = Counter()
    by_label = Counter()
    by_folder = Counter()
    samples = {"truncated": [], "struct_longer": [], "subst": [], "prefix_ok": []}
    len_deltas = []

    for r in recs:
        i = rid(r["header"])
        hit = I.get(i.lower()) if i else None
        if not hit:
            continue
        pdb, folder = hit
        aa = foldseek_aa(pdb)
        seq = r["sequence"]
        if aa is None:
            cat["foldseek_fail"] += 1
            continue
        aa = aa.upper()
        if aa == seq:
            cat["match"] += 1
            continue

        by_label[r["label"]] += 1
        by_folder[folder] += 1
        delta = len(seq) - len(aa)
        len_deltas.append(delta)

        if len(aa) < len(seq):
            # structure shorter than FASTA
            kind = "prefix_ok" if seq.startswith(aa) else "truncated"
        elif len(aa) > len(seq):
            kind = "struct_longer"
        else:
            kind = "subst"
        cat[kind] += 1
        if len(samples[kind]) < 6:
            # first diff position
            diff = next((k for k in range(min(len(aa), len(seq))) if aa[k] != seq[k]), min(len(aa), len(seq)))
            samples[kind].append(
                f"{r['label']}/{folder} {i}: struct={len(aa)} fasta={len(seq)} "
                f"d={delta} firstdiff@{diff} s='{seq[max(0,diff-3):diff+3]}' a='{aa[max(0,diff-3):diff+3]}'"
            )

    total_mm = sum(by_label.values())
    print(f"=== {total_mm} mismatches ===")
    print("category:", dict(cat))
    print("by_label:", dict(by_label))
    print("by_folder:", dict(by_folder))
    if len_deltas:
        import statistics as st
        pos = sum(1 for d in len_deltas if d > 0)
        neg = sum(1 for d in len_deltas if d < 0)
        zero = sum(1 for d in len_deltas if d == 0)
        print(f"len delta (fasta-struct): >0(struct shorter)={pos} <0(struct longer)={neg} =0={zero}")
        print(f"   median delta={st.median(len_deltas)} min={min(len_deltas)} max={max(len_deltas)}")
    for k, v in samples.items():
        if v:
            print(f"\n--- {k} samples ---")
            for s in v:
                print("  ", s)


if __name__ == "__main__":
    main()
