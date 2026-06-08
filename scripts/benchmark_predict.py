#!/usr/bin/env python3
"""Predict on a benchmark FASTA with the trained ESM2 + linear-head checkpoint.

Writes a prediction TSV in the exact format benchmark/score.py expects:

    protein_id, pred, prob_0, prob_1, prob_2     (pred = argmax, 0=nonRBP/1=TF/2=TSP)

Our model is a frozen-ESM2 + separate linear head (.pt), not a single HF
EsmForSequenceClassification, so benchmark/run_classifier.py can't load it —
this is the drop-in equivalent. Embeddings use the same pooling as training
(src/rbpdetect2/train.py / src/rbpdetect2/plm_embed.py).

Example:
    python scripts/benchmark_predict.py benchmark/inphared/data/benchmark.fasta \
        benchmark/inphared/predictions/rbpdetect_new.tsv
"""

import argparse
import csv
from pathlib import Path

import torch

from rbpdetect2.plm_embed import embed_sequences, load_plm, select_device, select_dtype

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = ROOT / "models" / "rbpdetect2_linear_facebook_esm2_t33_650M_UR50D.pt"


def parse_fasta(path: Path) -> list[dict[str, str]]:
    records, header, parts = [], None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    records.append({"id": header.split()[0], "seq": "".join(parts)})
                header, parts = line[1:], []
            elif line:
                parts.append(line)
    if header is not None:
        records.append({"id": header.split()[0], "seq": "".join(parts)})
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fasta", type=Path, help="Input benchmark FASTA")
    ap.add_argument("output_tsv", type=Path, help="Output prediction TSV")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-residues", type=int, default=1022)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    args = ap.parse_args()

    records = parse_fasta(args.fasta)
    if not records:
        raise SystemExit(f"No sequences in {args.fasta}")
    print(f"Loaded {len(records)} sequences from {args.fasta}")

    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)
    print(f"Device: {device}  dtype: {dtype}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    embed_dim = ckpt["embed_dim"]
    n_classes = len(ckpt["label2id"])
    plm, tokenizer, plm_dim = load_plm(ckpt["plm_model_name"], device, dtype)
    if plm_dim != embed_dim:
        raise SystemExit(f"PLM dim {plm_dim} != checkpoint embed_dim {embed_dim}")

    head = torch.nn.Linear(embed_dim, n_classes)
    head.load_state_dict({"weight": ckpt["model_state_dict"]["head.weight"],
                          "bias": ckpt["model_state_dict"]["head.bias"]})
    head.to(device).eval()

    embeddings = embed_sequences(
        plm, tokenizer, [r["seq"] for r in records], device,
        batch_size=args.batch_size, max_residues=args.max_residues,
    )
    with torch.inference_mode():
        X = torch.tensor(embeddings, dtype=torch.float32, device=device)
        probs = torch.softmax(head(X), dim=1).cpu().numpy()
    preds = probs.argmax(1)

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_tsv, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["protein_id", "pred"] + [f"prob_{k}" for k in range(n_classes)])
        for r, pred, prob in zip(records, preds, probs):
            w.writerow([r["id"], int(pred)] + [f"{p:.6f}" for p in prob])
    print(f"Wrote {len(records)} predictions to {args.output_tsv}")
    print(f"pred counts: nonRBP={int((preds==0).sum())} TF={int((preds==1).sum())} TSP={int((preds==2).sum())}")


if __name__ == "__main__":
    main()
