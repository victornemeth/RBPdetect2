# RBPdetect2 benchmark

Self-contained benchmark for phage receptor-binding protein (RBP) detection.
Drop a newly trained RBPdetect2 model in and get a side-by-side comparison
against the previous model and every competing tool — **no need to install or
re-run any of the other tools**, their predictions are precomputed and shipped
here.

## What's in the box

Two independent evaluation sets, each fully self-contained:

| Set            | Proteins | What it is |
|----------------|----------|------------|
| `inphared`     | 2,957    | Phage genomes deposited **after the Aug-2025 training cutoff** (temporal hold-out, deduplicated to <50% identity vs the training set). The primary benchmark. |
| `experimental` | 431      | Experimentally-validated proteins (UniProt ECO:0000269, PDB, BRENDA), tagged with an evidence tier (T1 direct assay → T3 structure-only). |

```
benchmark/
├── score.py                 # scores every tool, writes the comparison table
├── run_classifier.py        # runs a 3-class ESM model -> prediction TSV
├── requirements.txt
├── inphared/
│   ├── data/benchmark.tsv        # ground truth (protein_id, label, ...)
│   ├── data/benchmark.fasta      # sequences to predict on
│   ├── predictions/*.tsv         # precomputed baselines (see below)
│   └── results/                  # written by score.py
└── experimental/                 # same layout
```

**Label space:** `NEG` (negative), `TF` (tail fiber), `TSP` (tail spike),
`RBP` (generic receptor-binding protein). Binary RBP-positive = `label != NEG`.

**Precomputed baselines** (per set): `rbpdetect2` (the *previous* RBPdetect2),
`rbpdetect_v4` (PhageRBPdetect v4, Boeckaerts 2024), `phold`, `pharokka`,
`phanns` (inphared only), `blastp` (experimental only), `deposcope`,
`spikehunter`. DepoScope and SpikeHunter are tailspike-specialised — they are
scored on the TSP-only sub-task.

## Benchmark a new model

The model must be a 3-class HuggingFace `EsmForSequenceClassification`
(`0=Non-RBP, 1=Tail Fiber, 2=Tail Spike`) — same scheme as RBPdetect2.

```bash
pip install -r requirements.txt

# 1. Predict (writes the new-model slot; MODEL_DIR holds config.json + model.safetensors)
python run_classifier.py MODEL_DIR inphared/data/benchmark.fasta     inphared/predictions/rbpdetect_new.tsv
python run_classifier.py MODEL_DIR experimental/data/benchmark.fasta experimental/predictions/rbpdetect_new.tsv

# 2. Score (the rbpdetect_new.tsv slot is picked up automatically)
python score.py inphared
python score.py experimental
```

Each run prints the comparison and writes to `<set>/results/`:

- `metrics_overall.tsv` — precision / recall / specificity / F1 / MCC / accuracy per tool
- `recall_by_label.tsv` — recall on the TF, TSP and generic-RBP slices
- `metrics_by_tier.tsv` — per evidence tier (experimental set only)
- `rbpdetect_confusion_matrix.tsv` — 3-class confusion for the RBPdetect2 model
- `summary.md` — the same, as a markdown report ready to paste

The shipped `results/` are the baselines-only run; re-running after step 1 adds
the **RBPdetect2 (new, this work)** row next to **RBPdetect2 (previous)**.

## Reproducibility

Re-evaluating a new model is fully reproducible (deterministic predict + score,
pinned versions in `requirements.txt`). The datasets and baseline tool
predictions are shipped frozen — regenerating them from raw inputs needs the
external tools/DBs. See `PROVENANCE.md` for the full breakdown.

## Notes

- Only a `pred` column (and `Predicted_label` for SpikeHunter) is read from each
  prediction TSV; probability columns are ignored by scoring.
- Adding another tool = add one line to the `TOOLS` registry at the top of
  `score.py` and drop its `predictions/<name>.tsv`.
- Long sequences are truncated at the model's positional limit by
  `run_classifier.py`; all baseline tools were run at their own defaults.
