# rbpdetect2

Ternary classifier for bacteriophage **Receptor Binding Proteins** (RBPs): Tail Fiber (TF), Tail Spike Protein (TSP), or non-RBP. Uses frozen **ESMC-6B** embeddings + a trainable linear head.

## Requirements

| Dependency | How to get |
|---|---|
| Python 3.12 | auto-downloaded by uv |
| uv | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| mmseqs2 | `conda install -c bioconda mmseqs2` |
| CUDA 12+ drivers | system install |
| texlive-xetex *(optional, PDF export)* | `sudo apt install texlive-xetex texlive-fonts-recommended` |

**GPU:** ESMC-6B in bfloat16 requires ~12 GB VRAM. The notebook auto-selects the GPU with the most free memory. CPU inference works but is very slow.

## Setup

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create venv and install all Python deps
uv venv && uv pip install -e ".[dev]"

# 3. Install mmseqs2 (if not already present)
conda install -c bioconda mmseqs2

# 4. Log in to HuggingFace and accept the ESMC-6B license
#    → https://huggingface.co/biohub/ESMC-6B
huggingface-cli login

# 5. Register the kernel and launch JupyterLab
source .venv/bin/activate
python -m ipykernel install --user --name rbpdetect2 --display-name "rbpdetect2 (py3.12)"
jupyter lab
```

## Data

Raw data goes in `data/raw/` (not tracked by git). See `CLAUDE.md` for the full file list.

```bash
# Step 1 — combine raw FASTAs into processed files
python scripts/combine_fastas.py

# Step 2 — sequence diversity analysis (requires mmseqs2)
python scripts/diversity_analysis.py
```

Processed files written to `data/`: `tf.fasta` (778 seqs), `tsp.fasta` (205 seqs), `nonrbp.fasta` (1386 seqs).

## Training

Open `notebooks/train_classifier.ipynb` with the **rbpdetect2 (py3.12)** kernel.

The notebook:
1. Loads processed FASTAs
2. Runs cluster-aware train/val/test split (MMseqs2 @ 30% identity — prevents leakage)
3. Extracts ESMC-6B embeddings and caches them to `data/embeddings_cache.pt`
4. Plots PCA of embeddings coloured by class
5. Trains a linear classifier with class-weighted loss
6. Evaluates on the test set and saves the checkpoint to `models/`

Embedding extraction only runs once; subsequent runs load from cache.

## Repository layout

```
rbpdetect2/
├── data/
│   ├── raw/          # immutable source FASTAs (gitignored)
│   ├── tf.fasta      # 778 TF sequences
│   ├── tsp.fasta     # 205 TSP sequences
│   └── nonrbp.fasta  # 1386 non-RBP sequences
├── notebooks/
│   └── train_classifier.ipynb
├── scripts/
│   ├── combine_fastas.py      # merges raw sources, applies exclusion list
│   └── diversity_analysis.py  # MMseqs2 clustering + cross-group contamination
├── src/rbpdetect2/   # importable package (empty for now)
├── models/           # saved checkpoints (gitignored)
├── pyproject.toml
└── uv.lock
```
