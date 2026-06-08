# Benchmark results — inphared

Ground truth: **2957** proteins (TF=411, TSP=24, generic RBP=25, NEG=2497).

## Overall (binary)

Positive class = RBP (TF|TSP|generic RBP) for general tools; TSP-only for the depolymerase/tailspike-specialised tools (DepoScope, SpikeHunter).

| Tool | Task | TP | FP | TN | FN | Precision | Recall | Specificity | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| RBPdetect2 (previous) | RBP (TF|TSP|RBP) | 167 | 6 | 2491 | 293 | 0.965 | 0.363 | 0.998 | 0.528 | 0.557 |
| RBPdetect2 (new, this work) | RBP (TF|TSP|RBP) | 338 | 51 | 2446 | 122 | 0.869 | 0.735 | 0.980 | 0.796 | 0.766 |
| PhageRBPdetect v4 (Boeckaerts 2024) | RBP (TF|TSP|RBP) | 318 | 15 | 2482 | 142 | 0.955 | 0.691 | 0.994 | 0.802 | 0.786 |
| Phold (Bouras 2024) | RBP (TF|TSP|RBP) | 400 | 154 | 2343 | 60 | 0.722 | 0.870 | 0.938 | 0.789 | 0.750 |
| Pharokka (Bouras 2023) | RBP (TF|TSP|RBP) | 380 | 122 | 2375 | 80 | 0.757 | 0.826 | 0.951 | 0.790 | 0.750 |
| PhANNs (Cantu 2020) | RBP (TF|TSP|RBP) | 290 | 82 | 2415 | 170 | 0.780 | 0.630 | 0.967 | 0.697 | 0.653 |
| DepoScope (Concha-Eloko 2024) | TSP-only | 14 | 110 | 2823 | 10 | 0.113 | 0.583 | 0.962 | 0.189 | 0.244 |
| SpikeHunter (Yang 2024) | TSP-only | 14 | 22 | 2911 | 10 | 0.389 | 0.583 | 0.992 | 0.467 | 0.471 |

## Recall by ground-truth class

| Tool | Recall TF | Recall TSP | Recall generic-RBP |
|---|---|---|---|
| RBPdetect2 (previous) | 0.370 | 0.583 | 0.040 |
| RBPdetect2 (new, this work) | 0.742 | 0.708 | 0.640 |
| PhageRBPdetect v4 (Boeckaerts 2024) | 0.730 | 0.542 | 0.200 |
| Phold (Bouras 2024) | 0.883 | 0.833 | 0.680 |
| Pharokka (Bouras 2023) | 0.847 | 0.708 | 0.600 |
| PhANNs (Cantu 2020) | 0.647 | 0.625 | 0.360 |
| DepoScope (Concha-Eloko 2024) | 0.229 | 0.583 | 0.120 |
| SpikeHunter (Yang 2024) | 0.044 | 0.583 | 0.000 |

## RBPdetect2 3-class confusion matrix

Rows = true label, columns = predicted (0=NEG, 1=TF, 2=TSP).

```
pred     0    1    2   All
true                      
NEG   2446   18   33  2497
RBP      9   10    6    25
TF     106  186  119   411
TSP      7    3   14    24
All   2568  217  172  2957
```
