# Benchmark results — experimental

Ground truth: **431** proteins (TF=50, TSP=60, generic RBP=10, NEG=311).

## Overall (binary)

Positive class = RBP (TF|TSP|generic RBP) for general tools; TSP-only for the depolymerase/tailspike-specialised tools (DepoScope, SpikeHunter).

| Tool | Task | TP | FP | TN | FN | Precision | Recall | Specificity | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| RBPdetect2 (previous) | RBP (TF|TSP|RBP) | 81 | 1 | 310 | 39 | 0.988 | 0.675 | 0.997 | 0.802 | 0.767 |
| RBPdetect2 (new, this work) | RBP (TF|TSP|RBP) | 107 | 7 | 304 | 13 | 0.939 | 0.892 | 0.977 | 0.915 | 0.883 |
| PhageRBPdetect v4 (Boeckaerts 2024) | RBP (TF|TSP|RBP) | 95 | 5 | 306 | 25 | 0.950 | 0.792 | 0.984 | 0.864 | 0.824 |
| Phold (Bouras 2024) | RBP (TF|TSP|RBP) | 114 | 127 | 184 | 6 | 0.473 | 0.950 | 0.592 | 0.632 | 0.489 |
| Pharokka (Bouras 2023) | RBP (TF|TSP|RBP) | 103 | 115 | 196 | 17 | 0.472 | 0.858 | 0.630 | 0.609 | 0.438 |
| BLASTp baseline | RBP (TF|TSP|RBP) | 79 | 3 | 308 | 41 | 0.963 | 0.658 | 0.990 | 0.782 | 0.741 |
| DepoScope (Concha-Eloko 2024) | TSP-only | 53 | 19 | 352 | 7 | 0.736 | 0.883 | 0.949 | 0.803 | 0.772 |
| SpikeHunter (Yang 2024) | TSP-only | 50 | 8 | 363 | 10 | 0.862 | 0.833 | 0.978 | 0.847 | 0.823 |

## Recall by ground-truth class

| Tool | Recall TF | Recall TSP | Recall generic-RBP |
|---|---|---|---|
| RBPdetect2 (previous) | 0.540 | 0.883 | 0.100 |
| RBPdetect2 (new, this work) | 0.820 | 0.933 | 1.000 |
| PhageRBPdetect v4 (Boeckaerts 2024) | 0.700 | 0.867 | 0.800 |
| Phold (Bouras 2024) | 0.940 | 0.967 | 0.900 |
| Pharokka (Bouras 2023) | 0.860 | 0.833 | 1.000 |
| BLASTp baseline | 0.540 | 0.833 | 0.200 |
| DepoScope (Concha-Eloko 2024) | 0.320 | 0.883 | 0.200 |
| SpikeHunter (Yang 2024) | 0.160 | 0.833 | 0.000 |

## By evidence tier

| Tool | Tier | n | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| RBPdetect2 (previous) | T1 | 79 | 1.000 | 0.851 | 0.920 | 0.836 |
| RBPdetect2 (previous) | T2 | 80 | 1.000 | 0.375 | 0.545 | 0.569 |
| RBPdetect2 (previous) | T3 | 272 | 0.972 | 0.614 | 0.753 | 0.732 |
| RBPdetect2 (new, this work) | T1 | 79 | 0.957 | 0.936 | 0.946 | 0.870 |
| RBPdetect2 (new, this work) | T2 | 80 | 0.812 | 0.812 | 0.812 | 0.766 |
| RBPdetect2 (new, this work) | T3 | 272 | 0.962 | 0.877 | 0.917 | 0.898 |
| PhageRBPdetect v4 (Boeckaerts 2024) | T1 | 79 | 0.975 | 0.830 | 0.897 | 0.784 |
| PhageRBPdetect v4 (Boeckaerts 2024) | T2 | 80 | 0.867 | 0.812 | 0.839 | 0.801 |
| PhageRBPdetect v4 (Boeckaerts 2024) | T3 | 272 | 0.956 | 0.754 | 0.843 | 0.816 |
| Phold (Bouras 2024) | T1 | 79 | 0.763 | 0.957 | 0.849 | 0.587 |
| Phold (Bouras 2024) | T2 | 80 | 0.341 | 0.938 | 0.500 | 0.389 |
| Phold (Bouras 2024) | T3 | 272 | 0.391 | 0.947 | 0.554 | 0.453 |
| Pharokka (Bouras 2023) | T1 | 79 | 0.736 | 0.830 | 0.780 | 0.410 |
| Pharokka (Bouras 2023) | T2 | 80 | 0.356 | 1.000 | 0.525 | 0.441 |
| Pharokka (Bouras 2023) | T3 | 272 | 0.400 | 0.842 | 0.542 | 0.416 |
| BLASTp baseline | T1 | 79 | 0.974 | 0.787 | 0.871 | 0.743 |
| BLASTp baseline | T2 | 80 | 0.875 | 0.438 | 0.583 | 0.562 |
| BLASTp baseline | T3 | 272 | 0.972 | 0.614 | 0.753 | 0.732 |
| DepoScope (Concha-Eloko 2024) | T1 | 79 | 0.919 | 0.895 | 0.907 | 0.823 |
| DepoScope (Concha-Eloko 2024) | T2 | 80 | 0.800 | 1.000 | 0.889 | 0.889 |
| DepoScope (Concha-Eloko 2024) | T3 | 272 | 0.500 | 0.833 | 0.625 | 0.614 |
| SpikeHunter (Yang 2024) | T1 | 79 | 1.000 | 0.868 | 0.930 | 0.880 |
| SpikeHunter (Yang 2024) | T2 | 80 | 1.000 | 0.500 | 0.667 | 0.698 |
| SpikeHunter (Yang 2024) | T3 | 272 | 0.652 | 0.833 | 0.732 | 0.716 |

## RBPdetect2 3-class confusion matrix

Rows = true label, columns = predicted (0=NEG, 1=TF, 2=TSP).

```
pred    0   1   2  All
true                  
NEG   304   5   2  311
RBP     0   1   9   10
TF      9  25  16   50
TSP     4   2  54   60
All   317  33  81  431
```
