#!/usr/bin/env python3
"""Predict RBP class (nonRBP / TF / TSP) for protein sequences.

Loads the trained ESM2 + linear classifier, embeds each input sequence, and
writes a CSV with columns: id, seq, label, score.

  label   final call (nonRBP, TF or TSP)
  score   probability of that label under the model's softmax

A protein is called an RBP when P(TF) + P(TSP) >= --threshold (default 0.5),
and is then typed TF or TSP by whichever subtype probability is larger;
otherwise it is nonRBP. Lower the threshold for higher recall on RBPs.

With --export-fastas, predictions are also written as three FASTA files
(nonrbps.fasta, tfs.fasta, tsps.fasta) in the output directory.

Examples:
    python scripts/predict_cli.py input.fasta -o predictions.csv
    python scripts/predict_cli.py input.fasta -o pred.csv --export-fastas --threshold 0.7
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from rbpdetect2.plm_embed import embed_sequences, load_plm, select_device, select_dtype

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = ROOT / "models" / "rbpdetect2_linear_facebook_esm2_t33_650M_UR50D.pt"


def parse_fasta(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    header, parts = None, []
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


def write_fasta(path: Path, records: list[dict[str, str]]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(f">{r['id']}\n{r['seq']}\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("input", type=Path, help="Input FASTA file")
    ap.add_argument("-o", "--output", type=Path, default=Path("predictions.csv"),
                    help="Output CSV path (default: predictions.csv)")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT,
                    help="Trained model checkpoint (.pt)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="P(RBP)=P(TF)+P(TSP) cutoff to call a protein an RBP (default: 0.5)")
    ap.add_argument("--export-fastas", action="store_true",
                    help="Also write nonrbps/tfs/tsps FASTA files split by predicted label")
    ap.add_argument("--fasta-dir", type=Path, default=None,
                    help="Directory for exported FASTAs (default: alongside --output)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-residues", type=int, default=1022,
                    help="Residues per model call; longer proteins are chunked and pooled")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, or e.g. cuda:0")
    ap.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    args = ap.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold must be in [0, 1]")
    if not args.checkpoint.exists():
        raise SystemExit(f"Missing checkpoint {args.checkpoint}. Train the model first.")
    if not args.input.exists():
        raise SystemExit(f"Missing input FASTA {args.input}")

    records = parse_fasta(args.input)
    if not records:
        raise SystemExit(f"No sequences found in {args.input}")
    print(f"Loaded {len(records)} sequences from {args.input}")

    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)
    print(f"Device: {device}  dtype: {dtype}")

    # Load checkpoint (trusted, self-produced artifact).
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    id2label = {int(k): v for k, v in ckpt["id2label"].items()}
    label2id = ckpt["label2id"]
    embed_dim = ckpt["embed_dim"]
    plm_name = ckpt["plm_model_name"]

    plm, tokenizer, plm_dim = load_plm(plm_name, device, dtype)
    if plm_dim != embed_dim:
        raise SystemExit(f"PLM dim {plm_dim} != checkpoint embed_dim {embed_dim}")

    head = torch.nn.Linear(embed_dim, len(label2id))
    head.load_state_dict(
        {"weight": ckpt["model_state_dict"]["head.weight"],
         "bias": ckpt["model_state_dict"]["head.bias"]}
    )
    head.to(device).eval()

    embeddings = embed_sequences(
        plm, tokenizer, [r["seq"] for r in records], device,
        batch_size=args.batch_size, max_residues=args.max_residues,
    )

    with torch.inference_mode():
        X = torch.tensor(embeddings, dtype=torch.float32, device=device)
        probs = torch.softmax(head(X), dim=1).cpu().numpy()

    nonrbp_id = label2id["nonRBP"]
    tf_id, tsp_id = label2id["TF"], label2id["TSP"]
    p_rbp = probs[:, tf_id] + probs[:, tsp_id]

    labels, scores = [], []
    for i in range(len(records)):
        if p_rbp[i] >= args.threshold:
            sub = tf_id if probs[i, tf_id] >= probs[i, tsp_id] else tsp_id
            labels.append(id2label[sub])
            scores.append(float(probs[i, sub]))
        else:
            labels.append(id2label[nonrbp_id])
            scores.append(float(probs[i, nonrbp_id]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "seq", "label", "score"])
        for r, label, score in zip(records, labels, scores):
            writer.writerow([r["id"], r["seq"], label, f"{score:.4f}"])
    print(f"Wrote {len(records)} predictions to {args.output}")

    counts = {name: labels.count(name) for name in ("nonRBP", "TF", "TSP")}
    print(f"Predicted: nonRBP={counts['nonRBP']}  TF={counts['TF']}  TSP={counts['TSP']}")

    if args.export_fastas:
        fasta_dir = args.fasta_dir or args.output.parent
        fasta_dir.mkdir(parents=True, exist_ok=True)
        groups = {"nonRBP": [], "TF": [], "TSP": []}
        for r, label in zip(records, labels):
            groups[label].append(r)
        for label, fname in (("nonRBP", "nonrbps.fasta"), ("TF", "tfs.fasta"), ("TSP", "tsps.fasta")):
            write_fasta(fasta_dir / fname, groups[label])
            print(f"Wrote {len(groups[label])} seqs to {fasta_dir / fname}")


if __name__ == "__main__":
    main()
