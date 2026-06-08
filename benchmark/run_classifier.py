#!/usr/bin/env python
"""
Run the RBPdetect2 ESM-2 sequence classifier on a benchmark FASTA and write a
prediction TSV that ``score.py`` can read.

The model is a 3-class HuggingFace ``EsmForSequenceClassification``:
    0 = Non-RBP, 1 = Tail Fiber, 2 = Tail Spike

Output columns: protein_id, pred, prob_0, prob_1, prob_2

Typical use (from inside this benchmark folder):

    python run_classifier.py ../final_esm2_classifier inphared/data/benchmark.fasta \
        inphared/predictions/rbpdetect_new.tsv
    python run_classifier.py ../final_esm2_classifier experimental/data/benchmark.fasta \
        experimental/predictions/rbpdetect_new.tsv

Then score with:

    python score.py inphared
    python score.py experimental
"""
import argparse
import math
import sys

import pandas as pd
import torch
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_path", help="path to HF ESM classifier dir (config.json + model.safetensors)")
    p.add_argument("fasta", help="input protein FASTA")
    p.add_argument("output_tsv", help="output prediction TSV")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-len", type=int, default=2500)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)
    model.eval()

    # Clamp truncation to the model's positional limit (ESM reserves 2 for cls/eos).
    model_max = getattr(model.config, "max_position_embeddings", args.max_len + 2)
    eff_max = min(args.max_len, model_max - 2)
    print(f"Effective max_length: {eff_max} (model_max_position_embeddings={model_max})", file=sys.stderr)

    records = list(SeqIO.parse(args.fasta, "fasta"))
    print(f"Loaded {len(records)} sequences", file=sys.stderr)

    rows = []
    n_batches = math.ceil(len(records) / args.batch)
    for i in tqdm(range(0, len(records), args.batch), total=n_batches, desc="Predicting"):
        batch = records[i:i + args.batch]
        seqs = [str(r.seq) for r in batch]
        ids = [r.id for r in batch]
        try:
            enc = tokenizer(seqs, return_tensors="pt", truncation=True, padding=True,
                            max_length=eff_max).to(device)
            with torch.no_grad():
                logits = model(**enc).logits
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            for j, pid in enumerate(ids):
                row = {"protein_id": pid, "pred": int(preds[j])}
                for k in range(probs.shape[1]):
                    row[f"prob_{k}"] = float(probs[j, k])
                rows.append(row)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            for s, pid in zip(seqs, ids):  # one-by-one fallback
                enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=eff_max).to(device)
                with torch.no_grad():
                    logits = model(**enc).logits
                probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
                pred = int(logits.argmax(dim=-1).item())
                row = {"protein_id": pid, "pred": pred}
                for k in range(len(probs)):
                    row[f"prob_{k}"] = float(probs[k])
                rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(args.output_tsv, sep="\t", index=False)
    print(f"Wrote {len(out)} predictions to {args.output_tsv}", file=sys.stderr)


if __name__ == "__main__":
    main()
