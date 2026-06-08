# Clean-subset re-score — overlap with our training removed

Every tool re-scored on benchmark subsets after removing proteins that
overlap **our** training set at each dedup level. Overlap = exact match
(`id100`) or >=X% identity with >=80% of the benchmark
protein aligned (`id90/id70/id50`). Metric = RBP-binary F1
(TSP-only F1 for DepoScope/SpikeHunter).

⚠️ Dedup is vs OUR training only; baselines' training data is unavailable,
so their leakage is not removed. This removes our advantage, not theirs.

## inphared

Subset sizes (proteins kept / removed / positives):

| level | kept | removed | positives |
|---|---|---|---|
| all | 2957 | 0 | 460 |
| id100 | 2952 | 5 | 457 |
| id90 | 2918 | 39 | 436 |
| id70 | 2893 | 64 | 422 |
| id50 | 2832 | 125 | 387 |

F1 by tool across dedup levels:

| Tool | all | id100 | id90 | id70 | id50 |
|---|---|---|---|---|---|
| RBPdetect2 (previous) | 0.528 | 0.528 | 0.525 | 0.528 | 0.515 |
| RBPdetect2 (new, this work) | 0.796 | 0.796 | 0.793 | 0.785 | 0.767 |
| PhageRBPdetect v4 (Boeckaerts 2024) | 0.802 | 0.802 | 0.802 | 0.801 | 0.793 |
| Phold (Bouras 2024) | 0.789 | 0.788 | 0.783 | 0.781 | 0.768 |
| Pharokka (Bouras 2023) | 0.790 | 0.789 | 0.782 | 0.779 | 0.764 |
| PhANNs (Cantu 2020) | 0.697 | 0.696 | 0.694 | 0.690 | 0.684 |
| DepoScope (Concha-Eloko 2024) | 0.189 | 0.192 | 0.187 | 0.194 | 0.213 |
| SpikeHunter (Yang 2024) | 0.467 | 0.475 | 0.453 | 0.453 | 0.471 |

## experimental

Subset sizes (proteins kept / removed / positives):

| level | kept | removed | positives |
|---|---|---|---|
| all | 431 | 0 | 120 |
| id100 | 379 | 52 | 76 |
| id90 | 352 | 79 | 56 |
| id70 | 341 | 90 | 52 |
| id50 | 323 | 108 | 50 |

F1 by tool across dedup levels:

| Tool | all | id100 | id90 | id70 | id50 |
|---|---|---|---|---|---|
| RBPdetect2 (previous) | 0.802 | 0.695 | 0.593 | 0.548 | 0.563 |
| RBPdetect2 (new, this work) | 0.915 | 0.863 | 0.830 | 0.816 | 0.826 |
| PhageRBPdetect v4 (Boeckaerts 2024) | 0.864 | 0.800 | 0.742 | 0.719 | 0.714 |
| Phold (Bouras 2024) | 0.632 | 0.532 | 0.457 | 0.448 | 0.464 |
| Pharokka (Bouras 2023) | 0.609 | 0.510 | 0.431 | 0.419 | 0.434 |
| BLASTp baseline | 0.782 | 0.683 | 0.595 | 0.553 | 0.568 |
| DepoScope (Concha-Eloko 2024) | 0.803 | 0.635 | 0.457 | 0.438 | 0.438 |
| SpikeHunter (Yang 2024) | 0.847 | 0.706 | 0.593 | 0.560 | 0.560 |


## Takeaways

- **inphared barely moves** (new model F1 0.796 → 0.767 at id50): this set was
  already deduplicated to <50% vs the previous training set, so little of our
  overlap remains. New model stays competitive with the field — within ~0.03 F1
  of PhageRBPdetect v4 (0.793) and above Phold/Pharokka (~0.764) on the cleanest
  (id50) subset.
- **experimental drops for every tool**, because removing our-training overlap
  also strips out the well-characterised, easy proteins that all tools handle
  well. The new model still **leads at every dedup level** (id50 F1 0.826 vs v4
  0.714, SpikeHunter 0.560, BLASTp 0.568, Phold 0.464).
- **52 experimental proteins (12%) are byte-identical to our training** (id100) —
  genuine leakage. Even after removing them the new model holds 0.863, and the
  ranking is unchanged.
- **BLASTp collapses** (0.782 → 0.568): it blasts against *our* training set, so
  dedup against that training is exactly what it has no answer for — expected.
- Net: the new model's lead is **real, not a leakage artifact**. Quote the
  **id50** column as the conservative, fairest-available generalisation number:
  **F1 ≈ 0.77 (inphared)** and **≈ 0.83 (experimental)**.
