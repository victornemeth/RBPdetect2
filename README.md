# RBPdetect2

Classifier for bacteriophage **Receptor-Binding Proteins** (RBPs) — the
host-recognition proteins on phage tails. Given protein sequences, it assigns
each one of three classes:

- **TF** — Tail Fiber
- **TSP** — Tail Spike Protein
- **nonRBP** — anything else

RBPdetect2 is the **first tool to perform ternary RBP classification**
(TF / TSP / nonRBP), rather than the binary RBP-vs-not call of tools like
RBPdetect — it not only detects RBPs but distinguishes tail fibers from tail
spikes.

## How it works

Frozen **ESM2-650M** protein language model → mean-pooled residue embeddings
(1280-d) → a trainable linear head (3-class softmax). The PLM is never
fine-tuned; only the linear head is trained, so the model is small, fast, and
cheap to retrain. Training uses a **cluster-aware split** (MMseqs2 @ 30%
identity) so no cluster is shared across train/val/test, and class-balanced loss
to handle imbalance.

We also benchmarked larger and structure-aware language models (**ESMC-6B** and
**SaProt**) as the embedding backbone. They gave very similar classification
performance, so we chose ESM2-650M — the simplest, lightest, sequence-only
model that needs no structures.

## Getting started

A CUDA GPU is recommended (ESM2-650M in bf16 needs ~2.5 GB VRAM); CPU works but
is slow. ESM2 weights download automatically from Hugging Face (public, no
login).

Install [uv](https://docs.astral.sh/uv/) if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # macOS / Linux
# Windows (PowerShell): irm https://astral.sh/uv/install.ps1 | iex
```

Clone the repo and create the environment:

```bash
git clone https://github.com/victornemeth/RBPdetect2.git
cd RBPdetect2
uv sync                      # creates .venv and installs dependencies
```

### Predict

```bash
uv run rbpdetect2-predict input.fasta -o predictions.csv
```

Writes a CSV with `id, seq, label, score`. Options:

```bash
# also split sequences into nonrbps.fasta / tfs.fasta / tsps.fasta
uv run rbpdetect2-predict input.fasta -o pred.csv --export-fastas

# tune the RBP-detection threshold  (P(TF)+P(TSP) >= threshold -> RBP)
uv run rbpdetect2-predict input.fasta -o pred.csv --threshold 0.7
```

The trained checkpoint lives in `models/` and is **not** tracked in git
(gitignored). Produce it with the training script below, or download a release
(coming soon).

### Train

Needs `mmseqs2` on PATH (`conda install -c bioconda mmseqs2`) and the processed
FASTAs in `data/` (`tf.fasta`, `tsp.fasta`, `nonrbp.fasta`).

```bash
uv run rbpdetect2-train --epochs 50
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
rbpdetect2/
├── src/rbpdetect2/            importable package = the model + its tools
│   ├── model.py               LinearClassifier architecture
│   ├── train.py               training pipeline (rbpdetect2-train CLI)
│   ├── predict.py             inference pipeline (rbpdetect2-predict CLI)
│   ├── plm_embed.py           PLM embedding backends (ESMC, ESM2, SaProt)
│   ├── embedding_cli.py       shared CLI for embedding extraction
│   ├── benchmarking.py        frozen-embedding linear-probe utilities
│   └── benchmark_data.py      benchmark dataset loaders
├── scripts/                  one-off data-prep / analysis tools
│   ├── combine_fastas.py      build tf/tsp/nonrbp FASTAs
│   ├── diversity_analysis.py  mmseqs2 sequence-diversity / overlap check
│   ├── extract_esm{c,2}_embeddings.py, extract_saprot_embeddings.py
│   ├── build_saprot_sequences.py, diagnose_saprot_mismatch.py
│   ├── benchmark_trained_model.py, benchmark_predict.py
│   ├── check_benchmark_overlap.py, clean_benchmark_rescore.py
│   ├── per_class_tables.py, check_structures.py, filter_deposcope.py
├── notebooks/                training + embedding-benchmark notebooks
│   ├── train_classifier.ipynb
│   └── embedding_benchmark.ipynb
├── benchmark/                self-contained tool comparison (vs other tools)
│   ├── run_classifier.py, score.py
│   ├── experimental/          data, predictions/, results/ (experimental set)
│   ├── inphared/              data, predictions/, results/ (inphared set)
│   ├── RESULTS_*.md           per-class / clean-rescore / metric tables
│   └── PROVENANCE.md, README.md
├── benchmark_embeddings/     PLM embedding benchmark (ESMC-6B / ESM2 / SaProt)
│   ├── envs/                  per-model uv environments (esmc, esm2, saprot)
│   ├── embeddings/            precomputed embeddings (gitignored .npy + ids)
│   ├── splits/                stratified split_seed42 (committed for reuse)
│   ├── results/              metrics, confusion matrices, predictions
│   └── README.md
├── webapp/                   prediction web UI (app.py, index.html)
├── data/                     processed FASTAs + PCA plots (raw data gitignored)
├── config/, docs/           experiment configs, design notes
├── tests/                    test suite
├── pyproject.toml, uv.lock  project deps (managed with uv)
└── models/                  trained artifacts (gitignored)
```

## License

MIT — see [LICENSE](LICENSE).

## Citation

Coming soon.
