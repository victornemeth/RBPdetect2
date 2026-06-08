# Per-class & binary metric tables (by dedup level)

Precision / Recall / F1 / MCC. Binary = RBP detection (positive = TF|TSP|generic RBP); TF and TSP = one-vs-rest. Subtype derivation and caveats: see header of scripts/per_class_tables.py.

Dedup levels remove benchmark proteins overlapping **our** training: `id100` exact match, `id90/70/50` ≥X% identity over ≥80% of the protein.

# inphared

## inphared — all  (n=2957, TF=411, TSP=24, RBP-pos=460)

### Binary (RBP detection)

| Tool | Prec | Rec | F1 | MCC |
|---|---|---|---|---|
| RBPdetect2 (new, this work) | 0.869 | 0.735 | 0.796 | 0.766 |
| RBPdetect2 (previous) | 0.965 | 0.363 | 0.528 | 0.557 |
| PhageRBPdetect v4 (Boeckaerts 2024) | 0.955 | 0.691 | 0.802 | 0.786 |
| Phold (Bouras 2024) | 0.722 | 0.870 | 0.789 | 0.750 |
| Pharokka (Bouras 2023) | 0.757 | 0.826 | 0.790 | 0.750 |
| PhANNs (Cantu 2020) | 0.780 | 0.630 | 0.697 | 0.653 |
| DepoScope (Concha-Eloko 2024) (TSP-only) | 0.113 | 0.583 | 0.189 | 0.244 |
| SpikeHunter (Yang 2024) (TSP-only) | 0.389 | 0.583 | 0.467 | 0.471 |

### Per-class (TF / TSP, one-vs-rest)

| Tool | TF Prec | TF Rec | TF F1 | TF MCC | TSP Prec | TSP Rec | TSP F1 | TSP MCC |
|---|---|---|---|---|---|---|---|---|
| RBPdetect2 (new, this work) | 0.857 | 0.453 | 0.592 | 0.584 | 0.081 | 0.583 | 0.143 | 0.203 |
| RBPdetect2 (previous) | 0.970 | 0.311 | 0.471 | 0.519 | 0.341 | 0.583 | 0.431 | 0.441 |
| PhageRBPdetect v4 (Boeckaerts 2024) | 0.901 | 0.730 | 0.806 | 0.785 | 0.000 | 0.000 | 0.000 | 0.000 |
| Phold (Bouras 2024) | 0.668 | 0.876 | 0.758 | 0.722 | 0.733 | 0.458 | 0.564 | 0.577 |
| Pharokka (Bouras 2023) | 0.693 | 0.847 | 0.762 | 0.724 | 0.000 | 0.000 | 0.000 | 0.000 |
| PhANNs (Cantu 2020) | 0.715 | 0.647 | 0.679 | 0.632 | 0.000 | 0.000 | 0.000 | 0.000 |
| DepoScope (Concha-Eloko 2024) | — | — | — | — | 0.113 | 0.583 | 0.189 | 0.244 |
| SpikeHunter (Yang 2024) | — | — | — | — | 0.389 | 0.583 | 0.467 | 0.471 |

## inphared — id100  (n=2952, TF=408, TSP=24, RBP-pos=457)

### Binary (RBP detection)

| Tool | Prec | Rec | F1 | MCC |
|---|---|---|---|---|
| RBPdetect2 (new, this work) | 0.868 | 0.735 | 0.796 | 0.766 |
| RBPdetect2 (previous) | 0.965 | 0.363 | 0.528 | 0.557 |
| PhageRBPdetect v4 (Boeckaerts 2024) | 0.955 | 0.691 | 0.802 | 0.786 |
| Phold (Bouras 2024) | 0.721 | 0.869 | 0.788 | 0.749 |
| Pharokka (Bouras 2023) | 0.756 | 0.825 | 0.789 | 0.749 |
| PhANNs (Cantu 2020) | 0.778 | 0.630 | 0.696 | 0.653 |
| DepoScope (Concha-Eloko 2024) (TSP-only) | 0.115 | 0.583 | 0.192 | 0.247 |
| SpikeHunter (Yang 2024) (TSP-only) | 0.400 | 0.583 | 0.475 | 0.478 |

### Per-class (TF / TSP, one-vs-rest)

| Tool | TF Prec | TF Rec | TF F1 | TF MCC | TSP Prec | TSP Rec | TSP F1 | TSP MCC |
|---|---|---|---|---|---|---|---|---|
| RBPdetect2 (new, this work) | 0.856 | 0.453 | 0.593 | 0.585 | 0.082 | 0.583 | 0.144 | 0.204 |
| RBPdetect2 (previous) | 0.970 | 0.314 | 0.474 | 0.521 | 0.350 | 0.583 | 0.438 | 0.446 |
| PhageRBPdetect v4 (Boeckaerts 2024) | 0.900 | 0.730 | 0.806 | 0.785 | 0.000 | 0.000 | 0.000 | 0.000 |
| Phold (Bouras 2024) | 0.666 | 0.875 | 0.756 | 0.720 | 0.733 | 0.458 | 0.564 | 0.577 |
| Pharokka (Bouras 2023) | 0.691 | 0.846 | 0.761 | 0.723 | 0.000 | 0.000 | 0.000 | 0.000 |
| PhANNs (Cantu 2020) | 0.714 | 0.647 | 0.679 | 0.631 | 0.000 | 0.000 | 0.000 | 0.000 |
| DepoScope (Concha-Eloko 2024) | — | — | — | — | 0.115 | 0.583 | 0.192 | 0.247 |
| SpikeHunter (Yang 2024) | — | — | — | — | 0.400 | 0.583 | 0.475 | 0.478 |

## inphared — id90  (n=2918, TF=391, TSP=21, RBP-pos=436)

### Binary (RBP detection)

| Tool | Prec | Rec | F1 | MCC |
|---|---|---|---|---|
| RBPdetect2 (new, this work) | 0.864 | 0.732 | 0.793 | 0.763 |
| RBPdetect2 (previous) | 0.969 | 0.360 | 0.525 | 0.557 |
| PhageRBPdetect v4 (Boeckaerts 2024) | 0.953 | 0.693 | 0.802 | 0.787 |
| Phold (Bouras 2024) | 0.715 | 0.865 | 0.783 | 0.745 |
| Pharokka (Bouras 2023) | 0.751 | 0.817 | 0.782 | 0.743 |
| PhANNs (Cantu 2020) | 0.772 | 0.631 | 0.694 | 0.651 |
| DepoScope (Concha-Eloko 2024) (TSP-only) | 0.110 | 0.619 | 0.187 | 0.250 |
| SpikeHunter (Yang 2024) (TSP-only) | 0.375 | 0.571 | 0.453 | 0.458 |

### Per-class (TF / TSP, one-vs-rest)

| Tool | TF Prec | TF Rec | TF F1 | TF MCC | TSP Prec | TSP Rec | TSP F1 | TSP MCC |
|---|---|---|---|---|---|---|---|---|
| RBPdetect2 (new, this work) | 0.856 | 0.442 | 0.583 | 0.578 | 0.078 | 0.619 | 0.138 | 0.206 |
| RBPdetect2 (previous) | 0.968 | 0.309 | 0.469 | 0.518 | 0.324 | 0.571 | 0.414 | 0.425 |
| PhageRBPdetect v4 (Boeckaerts 2024) | 0.902 | 0.731 | 0.808 | 0.787 | 0.000 | 0.000 | 0.000 | 0.000 |
| Phold (Bouras 2024) | 0.661 | 0.870 | 0.751 | 0.716 | 0.692 | 0.429 | 0.529 | 0.542 |
| Pharokka (Bouras 2023) | 0.692 | 0.839 | 0.758 | 0.721 | 0.000 | 0.000 | 0.000 | 0.000 |
| PhANNs (Cantu 2020) | 0.711 | 0.647 | 0.677 | 0.631 | 0.000 | 0.000 | 0.000 | 0.000 |
| DepoScope (Concha-Eloko 2024) | — | — | — | — | 0.110 | 0.619 | 0.187 | 0.250 |
| SpikeHunter (Yang 2024) | — | — | — | — | 0.375 | 0.571 | 0.453 | 0.458 |

## inphared — id70  (n=2893, TF=379, TSP=21, RBP-pos=422)

### Binary (RBP detection)

| Tool | Prec | Rec | F1 | MCC |
|---|---|---|---|---|
| RBPdetect2 (new, this work) | 0.859 | 0.723 | 0.785 | 0.756 |
| RBPdetect2 (previous) | 0.968 | 0.363 | 0.528 | 0.560 |
| PhageRBPdetect v4 (Boeckaerts 2024) | 0.951 | 0.692 | 0.801 | 0.786 |
| Phold (Bouras 2024) | 0.712 | 0.865 | 0.781 | 0.744 |
| Pharokka (Bouras 2023) | 0.747 | 0.813 | 0.779 | 0.740 |
| PhANNs (Cantu 2020) | 0.766 | 0.628 | 0.690 | 0.647 |
| DepoScope (Concha-Eloko 2024) (TSP-only) | 0.115 | 0.619 | 0.194 | 0.256 |
| SpikeHunter (Yang 2024) (TSP-only) | 0.375 | 0.571 | 0.453 | 0.458 |

### Per-class (TF / TSP, one-vs-rest)

| Tool | TF Prec | TF Rec | TF F1 | TF MCC | TSP Prec | TSP Rec | TSP F1 | TSP MCC |
|---|---|---|---|---|---|---|---|---|
| RBPdetect2 (new, this work) | 0.859 | 0.449 | 0.589 | 0.584 | 0.083 | 0.619 | 0.146 | 0.213 |
| RBPdetect2 (previous) | 0.967 | 0.309 | 0.468 | 0.518 | 0.324 | 0.571 | 0.414 | 0.425 |
| PhageRBPdetect v4 (Boeckaerts 2024) | 0.899 | 0.728 | 0.805 | 0.784 | 0.000 | 0.000 | 0.000 | 0.000 |
| Phold (Bouras 2024) | 0.660 | 0.871 | 0.751 | 0.717 | 0.692 | 0.429 | 0.529 | 0.542 |
| Pharokka (Bouras 2023) | 0.691 | 0.836 | 0.757 | 0.720 | 0.000 | 0.000 | 0.000 | 0.000 |
| PhANNs (Cantu 2020) | 0.702 | 0.641 | 0.670 | 0.624 | 0.000 | 0.000 | 0.000 | 0.000 |
| DepoScope (Concha-Eloko 2024) | — | — | — | — | 0.115 | 0.619 | 0.194 | 0.256 |
| SpikeHunter (Yang 2024) | — | — | — | — | 0.375 | 0.571 | 0.453 | 0.458 |

## inphared — id50  (n=2832, TF=347, TSP=21, RBP-pos=387)

### Binary (RBP detection)

| Tool | Prec | Rec | F1 | MCC |
|---|---|---|---|---|
| RBPdetect2 (new, this work) | 0.845 | 0.703 | 0.767 | 0.738 |
| RBPdetect2 (previous) | 0.965 | 0.351 | 0.515 | 0.552 |
| PhageRBPdetect v4 (Boeckaerts 2024) | 0.946 | 0.682 | 0.793 | 0.779 |
| Phold (Bouras 2024) | 0.695 | 0.858 | 0.768 | 0.732 |
| Pharokka (Bouras 2023) | 0.731 | 0.801 | 0.764 | 0.726 |
| PhANNs (Cantu 2020) | 0.762 | 0.620 | 0.684 | 0.644 |
| DepoScope (Concha-Eloko 2024) (TSP-only) | 0.129 | 0.619 | 0.213 | 0.272 |
| SpikeHunter (Yang 2024) (TSP-only) | 0.400 | 0.571 | 0.471 | 0.473 |

### Per-class (TF / TSP, one-vs-rest)

| Tool | TF Prec | TF Rec | TF F1 | TF MCC | TSP Prec | TSP Rec | TSP F1 | TSP MCC |
|---|---|---|---|---|---|---|---|---|
| RBPdetect2 (new, this work) | 0.866 | 0.464 | 0.604 | 0.601 | 0.096 | 0.619 | 0.166 | 0.231 |
| RBPdetect2 (previous) | 0.962 | 0.294 | 0.450 | 0.505 | 0.343 | 0.571 | 0.429 | 0.437 |
| PhageRBPdetect v4 (Boeckaerts 2024) | 0.889 | 0.715 | 0.792 | 0.773 | 0.000 | 0.000 | 0.000 | 0.000 |
| Phold (Bouras 2024) | 0.645 | 0.865 | 0.739 | 0.706 | 0.692 | 0.429 | 0.529 | 0.542 |
| Pharokka (Bouras 2023) | 0.677 | 0.827 | 0.744 | 0.709 | 0.000 | 0.000 | 0.000 | 0.000 |
| PhANNs (Cantu 2020) | 0.695 | 0.631 | 0.662 | 0.618 | 0.000 | 0.000 | 0.000 | 0.000 |
| DepoScope (Concha-Eloko 2024) | — | — | — | — | 0.129 | 0.619 | 0.213 | 0.272 |
| SpikeHunter (Yang 2024) | — | — | — | — | 0.400 | 0.571 | 0.471 | 0.473 |

# experimental

## experimental — all  (n=431, TF=50, TSP=60, RBP-pos=120)

### Binary (RBP detection)

| Tool | Prec | Rec | F1 | MCC |
|---|---|---|---|---|
| RBPdetect2 (new, this work) | 0.939 | 0.892 | 0.915 | 0.883 |
| RBPdetect2 (previous) | 0.988 | 0.675 | 0.802 | 0.767 |
| PhageRBPdetect v4 (Boeckaerts 2024) | 0.950 | 0.792 | 0.864 | 0.824 |
| Phold (Bouras 2024) | 0.473 | 0.950 | 0.632 | 0.489 |
| Pharokka (Bouras 2023) | 0.472 | 0.858 | 0.609 | 0.438 |
| BLASTp baseline | 0.963 | 0.658 | 0.782 | 0.741 |
| DepoScope (Concha-Eloko 2024) (TSP-only) | 0.736 | 0.883 | 0.803 | 0.772 |
| SpikeHunter (Yang 2024) (TSP-only) | 0.862 | 0.833 | 0.847 | 0.823 |

### Per-class (TF / TSP, one-vs-rest)

| Tool | TF Prec | TF Rec | TF F1 | TF MCC | TSP Prec | TSP Rec | TSP F1 | TSP MCC |
|---|---|---|---|---|---|---|---|---|
| RBPdetect2 (new, this work) | 0.758 | 0.500 | 0.602 | 0.577 | 0.667 | 0.900 | 0.766 | 0.733 |
| RBPdetect2 (previous) | 0.900 | 0.360 | 0.514 | 0.540 | 0.839 | 0.867 | 0.852 | 0.828 |
| PhageRBPdetect v4 (Boeckaerts 2024) | 0.350 | 0.700 | 0.467 | 0.402 | 0.000 | 0.000 | 0.000 | 0.000 |
| Phold (Bouras 2024) | 0.205 | 0.900 | 0.333 | 0.282 | 0.619 | 0.217 | 0.321 | 0.314 |
| Pharokka (Bouras 2023) | 0.197 | 0.860 | 0.321 | 0.257 | 0.000 | 0.000 | 0.000 | 0.000 |
| BLASTp baseline | 0.792 | 0.380 | 0.514 | 0.512 | 0.845 | 0.817 | 0.831 | 0.804 |
| DepoScope (Concha-Eloko 2024) | — | — | — | — | 0.736 | 0.883 | 0.803 | 0.772 |
| SpikeHunter (Yang 2024) | — | — | — | — | 0.862 | 0.833 | 0.847 | 0.823 |

## experimental — id100  (n=379, TF=43, TSP=27, RBP-pos=76)

### Binary (RBP detection)

| Tool | Prec | Rec | F1 | MCC |
|---|---|---|---|---|
| RBPdetect2 (new, this work) | 0.900 | 0.829 | 0.863 | 0.831 |
| RBPdetect2 (previous) | 0.976 | 0.539 | 0.695 | 0.684 |
| PhageRBPdetect v4 (Boeckaerts 2024) | 0.915 | 0.711 | 0.800 | 0.766 |
| Phold (Bouras 2024) | 0.372 | 0.934 | 0.532 | 0.431 |
| Pharokka (Bouras 2023) | 0.368 | 0.829 | 0.510 | 0.380 |
| BLASTp baseline | 0.932 | 0.539 | 0.683 | 0.662 |
| DepoScope (Concha-Eloko 2024) (TSP-only) | 0.556 | 0.741 | 0.635 | 0.610 |
| SpikeHunter (Yang 2024) (TSP-only) | 0.750 | 0.667 | 0.706 | 0.686 |

### Per-class (TF / TSP, one-vs-rest)

| Tool | TF Prec | TF Rec | TF F1 | TF MCC | TSP Prec | TSP Rec | TSP F1 | TSP MCC |
|---|---|---|---|---|---|---|---|---|
| RBPdetect2 (new, this work) | 0.750 | 0.488 | 0.592 | 0.567 | 0.500 | 0.778 | 0.609 | 0.588 |
| RBPdetect2 (previous) | 1.000 | 0.326 | 0.491 | 0.547 | 0.714 | 0.741 | 0.727 | 0.706 |
| PhageRBPdetect v4 (Boeckaerts 2024) | 0.475 | 0.651 | 0.549 | 0.489 | 0.000 | 0.000 | 0.000 | 0.000 |
| Phold (Bouras 2024) | 0.216 | 0.884 | 0.347 | 0.301 | 0.467 | 0.259 | 0.333 | 0.312 |
| Pharokka (Bouras 2023) | 0.216 | 0.860 | 0.346 | 0.294 | 0.000 | 0.000 | 0.000 | 0.000 |
| BLASTp baseline | 0.789 | 0.349 | 0.484 | 0.490 | 0.720 | 0.667 | 0.692 | 0.670 |
| DepoScope (Concha-Eloko 2024) | — | — | — | — | 0.556 | 0.741 | 0.635 | 0.610 |
| SpikeHunter (Yang 2024) | — | — | — | — | 0.750 | 0.667 | 0.706 | 0.686 |

## experimental — id90  (n=352, TF=35, TSP=15, RBP-pos=56)

### Binary (RBP detection)

| Tool | Prec | Rec | F1 | MCC |
|---|---|---|---|---|
| RBPdetect2 (new, this work) | 0.880 | 0.786 | 0.830 | 0.802 |
| RBPdetect2 (previous) | 0.960 | 0.429 | 0.593 | 0.605 |
| PhageRBPdetect v4 (Boeckaerts 2024) | 0.878 | 0.643 | 0.742 | 0.714 |
| Phold (Bouras 2024) | 0.305 | 0.911 | 0.457 | 0.380 |
| Pharokka (Bouras 2023) | 0.297 | 0.786 | 0.431 | 0.322 |
| BLASTp baseline | 0.893 | 0.446 | 0.595 | 0.590 |
| DepoScope (Concha-Eloko 2024) (TSP-only) | 0.400 | 0.533 | 0.457 | 0.434 |
| SpikeHunter (Yang 2024) (TSP-only) | 0.667 | 0.533 | 0.593 | 0.580 |

### Per-class (TF / TSP, one-vs-rest)

| Tool | TF Prec | TF Rec | TF F1 | TF MCC | TSP Prec | TSP Rec | TSP F1 | TSP MCC |
|---|---|---|---|---|---|---|---|---|
| RBPdetect2 (new, this work) | 0.739 | 0.486 | 0.586 | 0.565 | 0.333 | 0.600 | 0.429 | 0.415 |
| RBPdetect2 (previous) | 1.000 | 0.314 | 0.478 | 0.541 | 0.571 | 0.533 | 0.552 | 0.533 |
| PhageRBPdetect v4 (Boeckaerts 2024) | 0.537 | 0.629 | 0.579 | 0.530 | 0.000 | 0.000 | 0.000 | 0.000 |
| Phold (Bouras 2024) | 0.197 | 0.886 | 0.323 | 0.294 | 0.300 | 0.200 | 0.240 | 0.218 |
| Pharokka (Bouras 2023) | 0.196 | 0.829 | 0.317 | 0.275 | 0.000 | 0.000 | 0.000 | 0.000 |
| BLASTp baseline | 0.733 | 0.314 | 0.440 | 0.447 | 0.538 | 0.467 | 0.500 | 0.481 |
| DepoScope (Concha-Eloko 2024) | — | — | — | — | 0.400 | 0.533 | 0.457 | 0.434 |
| SpikeHunter (Yang 2024) | — | — | — | — | 0.667 | 0.533 | 0.593 | 0.580 |

## experimental — id70  (n=341, TF=32, TSP=14, RBP-pos=52)

### Binary (RBP detection)

| Tool | Prec | Rec | F1 | MCC |
|---|---|---|---|---|
| RBPdetect2 (new, this work) | 0.870 | 0.769 | 0.816 | 0.788 |
| RBPdetect2 (previous) | 0.952 | 0.385 | 0.548 | 0.570 |
| PhageRBPdetect v4 (Boeckaerts 2024) | 0.865 | 0.615 | 0.719 | 0.691 |
| Phold (Bouras 2024) | 0.297 | 0.904 | 0.448 | 0.375 |
| Pharokka (Bouras 2023) | 0.288 | 0.769 | 0.419 | 0.312 |
| BLASTp baseline | 0.875 | 0.404 | 0.553 | 0.553 |
| DepoScope (Concha-Eloko 2024) (TSP-only) | 0.389 | 0.500 | 0.438 | 0.414 |
| SpikeHunter (Yang 2024) (TSP-only) | 0.636 | 0.500 | 0.560 | 0.548 |

### Per-class (TF / TSP, one-vs-rest)

| Tool | TF Prec | TF Rec | TF F1 | TF MCC | TSP Prec | TSP Rec | TSP F1 | TSP MCC |
|---|---|---|---|---|---|---|---|---|
| RBPdetect2 (new, this work) | 0.700 | 0.438 | 0.538 | 0.519 | 0.308 | 0.571 | 0.400 | 0.386 |
| RBPdetect2 (previous) | 1.000 | 0.250 | 0.400 | 0.482 | 0.538 | 0.500 | 0.519 | 0.499 |
| PhageRBPdetect v4 (Boeckaerts 2024) | 0.514 | 0.594 | 0.551 | 0.502 | 0.000 | 0.000 | 0.000 | 0.000 |
| Phold (Bouras 2024) | 0.188 | 0.875 | 0.309 | 0.284 | 0.222 | 0.143 | 0.174 | 0.150 |
| Pharokka (Bouras 2023) | 0.187 | 0.812 | 0.304 | 0.265 | 0.000 | 0.000 | 0.000 | 0.000 |
| BLASTp baseline | 0.692 | 0.281 | 0.400 | 0.409 | 0.545 | 0.429 | 0.480 | 0.464 |
| DepoScope (Concha-Eloko 2024) | — | — | — | — | 0.389 | 0.500 | 0.438 | 0.414 |
| SpikeHunter (Yang 2024) | — | — | — | — | 0.636 | 0.500 | 0.560 | 0.548 |

## experimental — id50  (n=323, TF=32, TSP=14, RBP-pos=50)

### Binary (RBP detection)

| Tool | Prec | Rec | F1 | MCC |
|---|---|---|---|---|
| RBPdetect2 (new, this work) | 0.905 | 0.760 | 0.826 | 0.802 |
| RBPdetect2 (previous) | 0.952 | 0.400 | 0.563 | 0.581 |
| PhageRBPdetect v4 (Boeckaerts 2024) | 0.882 | 0.600 | 0.714 | 0.690 |
| Phold (Bouras 2024) | 0.312 | 0.900 | 0.464 | 0.391 |
| Pharokka (Bouras 2023) | 0.304 | 0.760 | 0.434 | 0.328 |
| BLASTp baseline | 0.875 | 0.420 | 0.568 | 0.564 |
| DepoScope (Concha-Eloko 2024) (TSP-only) | 0.389 | 0.500 | 0.438 | 0.412 |
| SpikeHunter (Yang 2024) (TSP-only) | 0.636 | 0.500 | 0.560 | 0.547 |

### Per-class (TF / TSP, one-vs-rest)

| Tool | TF Prec | TF Rec | TF F1 | TF MCC | TSP Prec | TSP Rec | TSP F1 | TSP MCC |
|---|---|---|---|---|---|---|---|---|
| RBPdetect2 (new, this work) | 0.737 | 0.438 | 0.549 | 0.534 | 0.348 | 0.571 | 0.432 | 0.414 |
| RBPdetect2 (previous) | 1.000 | 0.250 | 0.400 | 0.481 | 0.538 | 0.500 | 0.519 | 0.498 |
| PhageRBPdetect v4 (Boeckaerts 2024) | 0.559 | 0.594 | 0.576 | 0.528 | 0.000 | 0.000 | 0.000 | 0.000 |
| Phold (Bouras 2024) | 0.207 | 0.875 | 0.335 | 0.307 | 0.222 | 0.143 | 0.174 | 0.149 |
| Pharokka (Bouras 2023) | 0.208 | 0.812 | 0.331 | 0.290 | 0.000 | 0.000 | 0.000 | 0.000 |
| BLASTp baseline | 0.692 | 0.281 | 0.400 | 0.407 | 0.545 | 0.429 | 0.480 | 0.463 |
| DepoScope (Concha-Eloko 2024) | — | — | — | — | 0.389 | 0.500 | 0.438 | 0.412 |
| SpikeHunter (Yang 2024) | — | — | — | — | 0.636 | 0.500 | 0.560 | 0.547 |

