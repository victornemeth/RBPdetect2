# RBPdetect2 (new) — benchmark log

Model: frozen **ESM2-650M** + linear head
(`models/rbpdetect2_linear_facebook_esm2_t33_650M_UR50D.pt`).
Predictions: `scripts/benchmark_predict.py` (argmax 0=nonRBP/1=TF/2=TSP) →
`<set>/predictions/rbpdetect_new.tsv`. Scored with `benchmark/score.py`.
Overlap: `scripts/check_benchmark_overlap.py` (MMseqs2 vs training tf/tsp/nonrbp).

## Headline — RBP binary (positive = TF|TSP|generic RBP)

| Set | Tool | Precision | Recall | F1 | MCC | Acc |
|---|---|---|---|---|---|---|
| inphared | **RBPdetect2 (new)** | 0.869 | 0.735 | **0.796** | 0.766 | 0.941 |
| inphared | RBPdetect2 (previous) | 0.965 | 0.363 | 0.528 | 0.557 | 0.899 |
| inphared | PhageRBPdetect v4 | 0.955 | 0.691 | 0.802 | 0.786 | 0.947 |
| inphared | Phold | 0.722 | 0.870 | 0.789 | 0.750 | 0.928 |
| inphared | Pharokka | 0.757 | 0.826 | 0.790 | 0.750 | 0.932 |
| inphared | PhANNs | 0.780 | 0.630 | 0.697 | 0.653 | 0.915 |
| experimental | **RBPdetect2 (new)** | 0.939 | 0.892 | **0.915** | 0.883 | 0.954 |
| experimental | RBPdetect2 (previous) | 0.988 | 0.675 | 0.802 | 0.767 | 0.907 |
| experimental | PhageRBPdetect v4 | 0.950 | 0.792 | 0.864 | 0.824 | 0.930 |
| experimental | Phold | 0.473 | 0.950 | 0.632 | 0.489 | 0.691 |
| experimental | BLASTp | 0.963 | 0.658 | 0.782 | 0.741 | 0.898 |

Full tables (incl. DepoScope/SpikeHunter TSP-only, per-tier, confusion):
`benchmark/<set>/results/summary.md`.

### Takeaways
- New model **beats the previous RBPdetect2** on both sets: F1 0.528→0.796
  (inphared), 0.802→0.915 (experimental). Driver is recall — the old model was
  very precise but missed most RBPs (recall 0.36 / 0.68); the new one recalls
  0.74 / 0.89 at small precision cost.
- vs external tools: roughly **on par with PhageRBPdetect v4 and Phold/Pharokka**
  on inphared (F1 ~0.79–0.80), and **best overall on the experimental set**
  (F1 0.915), beating v4 (0.864) and BLASTp (0.782).
- Recall by class is balanced (inphared TF 0.74 / TSP 0.71 / RBP 0.64), unlike the
  old model which barely caught generic RBP (0.04).

## ⚠️ Training-set overlap (leakage check)

MMseqs2 best-hit of each benchmark protein vs the combined training set
(3153 seqs). "Near-dup" = fident ≥95% with ≥80% mutual coverage; "homolog" =
fident ≥50% with ≥50% mutual coverage.

| Set | proteins | exact id | near-dup (≥95%) | homolog (≥50%) |
|---|---|---|---|---|
| inphared | 3058 | 1 | 70 (2.3%) | 242 (7.9%) |
| experimental | 431 | 0 | 68 (15.8%) | 109 (25.3%) |

Per-protein detail: `<set>/results/training_overlap.tsv`.

inphared was deduplicated to <50% identity vs the *previous* training set; the
current training set differs, so a small overlap tail reappears. The
**experimental set was not deduplicated** — a quarter of it is homologous to
training data, and ~16% are near-duplicates.

### Overlap inflates the scores — new-model F1 by subset

| Set | subset | n | pos | F1 | recall | precision |
|---|---|---|---|---|---|---|
| inphared | all | 2957 | 460 | 0.796 | 0.735 | 0.869 |
| inphared | homolog to train (≥50%) | 146 | 76 | 0.966 | 0.947 | 0.986 |
| inphared | **clean (<50% to train)** | 2811 | 384 | **0.760** | 0.693 | 0.842 |
| experimental | all | 431 | 120 | 0.915 | 0.892 | 0.939 |
| experimental | homolog to train (≥50%) | 109 | 71 | 0.972 | 0.986 | 0.959 |
| experimental | **clean (<50% to train)** | 322 | 49 | **0.822** | 0.755 | 0.902 |

**Honest generalisation numbers are the clean rows:** F1 **0.760** (inphared,
overlap effect small at −0.036) and F1 **0.822** (experimental, overlap effect
large at −0.093 — the experimental headline is meaningfully inflated by
non-deduplicated homologs/near-dups). Quote the clean-subset F1 when claiming
generalisation, especially on the experimental set.
