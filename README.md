# RBPdetect2

Classifier for bacteriophage **Receptor-Binding Proteins** (RBPs) — the
host-recognition proteins on phage tails. Given protein sequences, it assigns
each one of three classes:

- **TF** — Tail Fiber
- **TSP** — Tail Spike Protein
- **nonRBP** — anything else

## How it works

Frozen **ESM2-650M** protein language model → mean-pooled residue embeddings
(1280-d) → a trainable linear head (3-class softmax). The PLM is never
fine-tuned; only the linear head is trained, so the model is small, fast, and
cheap to retrain. Training uses a **cluster-aware split** (MMseqs2 @ 30%
identity) so no cluster is shared across train/val/test, and class-balanced loss
to handle imbalance.

## Getting started

Requires [uv](https://docs.astral.sh/uv/). A CUDA GPU is recommended (ESM2-650M
in bf16 needs ~2.5 GB VRAM); CPU works but is slow. ESM2 weights download
automatically from Hugging Face (public, no login).

```bash
uv sync                      # create the environment
```

### Predict

```bash
uv run python scripts/predict_cli.py input.fasta -o predictions.csv
```

Writes a CSV with `id, seq, label, score`. Options:

```bash
# also split sequences into nonrbps.fasta / tfs.fasta / tsps.fasta
uv run python scripts/predict_cli.py input.fasta -o pred.csv --export-fastas

# tune the RBP-detection threshold  (P(TF)+P(TSP) >= threshold -> RBP)
uv run python scripts/predict_cli.py input.fasta -o pred.csv --threshold 0.7
```

The trained checkpoint lives in `models/` and is **not** tracked in git
(gitignored). Produce it with the training script below, or download a release
(coming soon).

### Train

Needs `mmseqs2` on PATH (`conda install -c bioconda mmseqs2`) and the processed
FASTAs in `data/` (`tf.fasta`, `tsp.fasta`, `nonrbp.fasta`).

```bash
uv run python scripts/train_classifier.py --epochs 50
```

Saves `models/rbpdetect2_linear_<plm>.pt`. The same pipeline, with exploration
and plots, is in `notebooks/train_classifier.ipynb`.

## Benchmark

Evaluated against published tools on two held-out sets: **inphared** (phage
genomes deposited after the training cutoff) and **experimental**
(experimentally-validated proteins). Binary RBP detection (TF|TSP = positive):

| Tool | inphared F1 | experimental F1 |
|---|---|---|
| **RBPdetect2 (this work)** | **0.796** | **0.915** |
| PhageRBPdetect v4 (Boeckaerts 2024) | 0.802 | 0.864 |
| Phold (Bouras 2024) | 0.789 | 0.632 |
| Pharokka (Bouras 2023) | 0.790 | 0.609 |
| RBPdetect2 (previous) | 0.528 | 0.802 |

RBPdetect2 is competitive with the best general tools on inphared and the
strongest on the experimental set. Because benchmark proteins can overlap the
training set, results are also reported after removing proteins ≥50% identical
to training (conservative): F1 **0.767** (inphared) / **0.826** (experimental),
still leading. Full per-class, per-tool and overlap-controlled tables:

- `benchmark/RESULTS_new_model.md` — headline + training-overlap (leakage) check
- `benchmark/RESULTS_clean_rescore.md` — every tool re-scored after dedup
- `benchmark/RESULTS_metric_tables.md` — per-class (TF/TSP) and binary tables
- `benchmark/README.md` — how to re-run the benchmark

## Repository layout

```
src/rbpdetect2/   importable package (embedding + benchmark helpers)
scripts/          CLI tools (train, predict, benchmark, overlap check)
notebooks/        training + embedding-benchmark notebooks
benchmark/        self-contained tool comparison (data + precomputed baselines)
data/             processed FASTAs (raw data and models/ are gitignored)
tests/            test suite
```

## License

MIT — see [LICENSE](LICENSE).

## Citation

Coming soon.
