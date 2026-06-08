# Frozen embedding benchmark

This benchmark compares frozen ESMC-6B, ESM2-650M, and SaProt-650M embeddings
using the same cluster-aware split and the same balanced multinomial logistic
regression linear probe.

Each extractor writes a portable bundle:

```text
benchmark_embeddings/embeddings/<model>/
├── embeddings.npy
├── ids.tsv
└── metadata.json
```

The bundles are ignored by git because they are large. `metadata.json` records
the exact dataset hash, resolved model revision, pooling policy, runtime, and
software versions.

## Create environments

```bash
uv sync --project benchmark_embeddings/envs/esmc
uv sync --project benchmark_embeddings/envs/esm2
uv sync --project benchmark_embeddings/envs/saprot
```

Commit the generated `uv.lock` file in each environment before reporting final
benchmark results.

## Extract embeddings

All models use final-layer residue embeddings, mean pooling, and the same
default maximum chunk size of 1022 residues. Longer proteins are split into
non-overlapping chunks and the chunk means are combined using residue-count
weights.

```bash
uv run --project benchmark_embeddings/envs/esmc \
  python scripts/extract_esmc_embeddings.py

uv run --project benchmark_embeddings/envs/esm2 \
  python scripts/extract_esm2_embeddings.py

uv run --project benchmark_embeddings/envs/saprot \
  python scripts/extract_saprot_embeddings.py \
  --sa-sequences data/saprot_sequences.tsv
```

For final runs, pass `--revision <commit-hash>` to every extractor.

SaProt requires a tab-separated structure-aware sequence table:

```text
id	sa_sequence
protein_1	M#Ev...
```

`sa_sequence` must be the interleaved amino-acid and Foldseek 3Di sequence used
by SaProt. Its amino-acid characters, at positions `0, 2, 4, ...`, must match
the corresponding processed FASTA sequence exactly.

## Analyze

Open `notebooks/embedding_benchmark.ipynb` in the root rbpdetect2 environment.
The notebook:

1. Loads and validates all embedding bundles.
2. Creates or loads one all-sequence MMseqs2 cluster-aware split.
3. Tunes a balanced logistic-regression linear probe on validation macro-F1.
4. Evaluates the test set and writes metrics and predictions.
5. Produces a main benchmark figure with macro-F1 confidence intervals, PCA
   projections, and per-class F1 scores.

The split file in `benchmark_embeddings/splits/` should be committed so every model and
future rerun uses exactly the same held-out proteins.

