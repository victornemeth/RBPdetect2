<div align="center">

# RBPdetect2

## Identify Tailspike and Tail Fiber Proteins in Bacteriophages

<img src="./RBPdetect2.png" alt="Logo" width="300"/>

A **fast**, **easy-to-use** tool for identifying phage **receptor binding proteins (RBPs)** — including **tailspikes** and **tail fibers** — built with ESM2.

</div>

---

## Overview

**RBPdetect2** classifies bacteriophage proteins into one of three categories:

| Label       | Value | Description                             |
|-------------|-------|-----------------------------------------|
| Non-RBP     | **0** | Not a receptor-binding protein          |
| Tail Fiber  | **1** | A phage tail fiber protein              |
| Tail Spike  | **2** | A phage tail spike protein              |

---

## Features

- **Fast**: Parallelized batch processing
- **Accurate**: Powered by ESM2 protein language models
- **Simple**: Single-command execution
- **Flexible**: Works with standard FASTA files

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/victornemeth/RBPdetect2.git
cd RBPdetect2
pip install -r requirements.txt
```

## Quick Start
Run RBP classification on a FASTA file:

```bash
python inference_parallel.py ./final_esm2_classifier sequences.fasta -o output.tsv -b 64

```
- `-o`: Output file path (.tsv)

- `-b`: Batch size (default: 64)

## Optional: Clean Protein IDs
If your protein IDs contain ``_truncated_from_`` annotations, you can remove them with:
```bash
awk 'BEGIN{FS=OFS="\t"} {sub(/_truncated_from_.*/, "", $1); print}' output.tsv > output.cleaned.tsv
```
## Citation
Comming soon

## License
[MIT License](https://opensource.org/license/mit)