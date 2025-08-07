# RBPdetect2

RBPdetect2 is a tool for classifying bacteriophage proteins into one of three categories:

- **Non-RBP** (0)
- **Tail Fiber (TF)** (1)
- **Tail Spike Protein (TSP)** (2)

---

## Classification Labels

| Label       | Value | Description                             |
|:-----------:|:-----:|:---------------------------------------:|
| Non-RBP     | **0** | Not a receptor-binding protein         |
| Tail Fiber  | **1** | A phage tail fiber protein             |
| Tail Spike  | **2** | A phage tail spike protein             |

---

## Quick Start

```bash
# Example usage:
python inference_parallel.py ./final_esm2_classifier sequences.fasta -o output.tsv -b 64
