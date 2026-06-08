#!/usr/bin/env python3
"""Train the RBP classifier: frozen ESM2-650M embeddings + linear head.

CLI port of notebooks/train_classifier.ipynb (training pipeline only, no
plotting/exploration). Steps:

  1. Load processed FASTAs (tf / tsp / nonrbp)
  2. Cluster-aware stratified train/val/test split (MMseqs2 @ 30% identity)
  3. Extract & cache ESM2 embeddings
  4. Train a linear classifier with class-balanced loss
  5. Evaluate on the held-out test set
  6. Save the checkpoint (consumed by scripts/predict_cli.py)

Example:
    python scripts/train_classifier.py --epochs 50
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from rbpdetect2.plm_embed import embed_sequences, load_plm, select_device, select_dtype

ROOT = Path(__file__).resolve().parent.parent

LABEL2ID = {"nonRBP": 0, "TF": 1, "TSP": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def parse_fasta(path: Path, label: str) -> list[dict]:
    records, header, parts = [], None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    records.append({"id": header.split()[0], "sequence": "".join(parts), "label": label})
                header, parts = line[1:], []
            elif line:
                parts.append(line)
    if header is not None:
        records.append({"id": header.split()[0], "sequence": "".join(parts), "label": label})
    return records


def mmseqs_cluster_all(df: pd.DataFrame, min_id: float, cov: float = 0.8) -> dict[str, str]:
    """Return {seq_id: cluster_rep} for all sequences via MMseqs2 easy-cluster."""
    if not shutil.which("mmseqs"):
        raise RuntimeError("mmseqs not in PATH")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fasta = tmp / "all.fasta"
        with open(fasta, "w") as f:
            for _, row in df.iterrows():
                f.write(f">{row['id']}\n{row['sequence']}\n")
        prefix = tmp / "clust"
        subprocess.run([
            "mmseqs", "easy-cluster", str(fasta), str(prefix), str(tmp / "mmseqs_tmp"),
            "--min-seq-id", str(min_id), "-c", str(cov), "--cov-mode", "0",
            "--cluster-mode", "2", "-v", "0",
        ], check=True)
        member_to_rep = {}
        with open(str(prefix) + "_cluster.tsv") as f:
            for line in f:
                rep, member = line.strip().split("\t")
                member_to_rep[member] = rep
    return member_to_rep


def cluster_aware_split(
    df: pd.DataFrame, member_to_rep: dict[str, str],
    val_frac: float, test_frac: float, seed: int,
) -> pd.DataFrame:
    """Assign each sequence to train/val/test; whole clusters stay in one split."""
    df = df.copy()
    df["cluster"] = df["id"].map(member_to_rep)

    clust_info = (
        df.groupby("cluster")["label"]
        .agg(lambda s: s.value_counts().index[0])
        .reset_index().rename(columns={"label": "dominant_label"})
    )
    clust_info["size"] = df.groupby("cluster").size().values

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac + test_frac, random_state=seed)
    train_idx, holdout_idx = next(sss.split(clust_info["cluster"].values, clust_info["dominant_label"].values))

    holdout_info = clust_info.iloc[holdout_idx]
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=test_frac / (val_frac + test_frac), random_state=seed
    )
    val_idx, test_idx = next(
        sss2.split(holdout_info["cluster"].values, holdout_info["dominant_label"].values)
    )

    train_clusts = set(clust_info.iloc[train_idx]["cluster"])
    val_clusts = set(holdout_info.iloc[val_idx]["cluster"])

    def assign(c: str) -> str:
        if c in train_clusts:
            return "train"
        if c in val_clusts:
            return "val"
        return "test"

    df["split"] = df["cluster"].map(assign)
    return df


class LinearClassifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.head = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.head(x)


def run_epoch(loader, model, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models")
    ap.add_argument("--model-name", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--cache", type=Path, default=None,
                    help="Embedding cache .pt (default: <data-dir>/embeddings_cache_esm2.pt)")
    ap.add_argument("--use-cache", action="store_true", help="Reuse the embedding cache if present")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--cluster-id", type=float, default=0.3, help="MMseqs2 identity threshold")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--embed-batch-size", type=int, default=8)
    ap.add_argument("--max-residues", type=int, default=1022)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache or args.data_dir / "embeddings_cache_esm2.pt"

    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Device: {device}  ({name})")

    # 1. Load data
    df = pd.DataFrame(
        parse_fasta(args.data_dir / "tf.fasta", "TF")
        + parse_fasta(args.data_dir / "tsp.fasta", "TSP")
        + parse_fasta(args.data_dir / "nonrbp.fasta", "nonRBP")
    )
    df["label_id"] = df["label"].map(LABEL2ID)
    print(df["label"].value_counts().to_string())

    # 2. Cluster-aware split
    print(f"\nClustering all sequences with MMseqs2 @ id={args.cluster_id} ...")
    member_to_rep = mmseqs_cluster_all(df, min_id=args.cluster_id)
    df = cluster_aware_split(df, member_to_rep, args.val_frac, args.test_frac, args.seed)
    print("\nSplit counts:")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())

    # 3. Embeddings
    if args.use_cache and cache_path.exists():
        print(f"\nLoading cached embeddings from {cache_path}")
        cache = torch.load(cache_path, weights_only=False)
        id2emb = dict(zip(cache["ids"], cache["embeddings"]))
    else:
        print(f"\nExtracting embeddings for {len(df)} sequences ...")
        plm, tokenizer, embed_dim = load_plm(args.model_name, device, dtype)
        embeddings = embed_sequences(
            plm, tokenizer, df["sequence"].tolist(), device,
            batch_size=args.embed_batch_size, max_residues=args.max_residues,
        )
        ids = df["id"].tolist()
        torch.save({"ids": ids, "embeddings": embeddings}, cache_path)
        id2emb = dict(zip(ids, embeddings))
        print(f"Saved embeddings to {cache_path}")
        del plm
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df["embedding"] = df["id"].map(id2emb)
    embed_dim = int(df["embedding"].iloc[0].shape[0])
    print(f"Embedding dim: {embed_dim}")

    # 4. Train
    def make_loader(split: str, shuffle: bool) -> DataLoader:
        sub = df[df["split"] == split]
        X = torch.tensor(np.stack(sub["embedding"].values), dtype=torch.float32)
        y = torch.tensor(sub["label_id"].values, dtype=torch.long)
        return DataLoader(TensorDataset(X, y), batch_size=args.batch_size, shuffle=shuffle)

    train_loader = make_loader("train", True)
    val_loader = make_loader("val", False)
    test_loader = make_loader("test", False)

    train_labels = df[df["split"] == "train"]["label_id"].values
    class_weights = compute_class_weight("balanced", classes=np.arange(len(LABEL2ID)), y=train_labels)
    print(f"Class weights: { {ID2LABEL[i]: round(w, 3) for i, w in enumerate(class_weights)} }")

    model = LinearClassifier(embed_dim, len(LABEL2ID)).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc, best_state = 0.0, None
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(train_loader, model, criterion, device, optimizer)
        vl_loss, vl_acc = run_epoch(val_loader, model, criterion, device)
        scheduler.step()
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | tr_loss {tr_loss:.4f} tr_acc {tr_acc:.3f} | "
                  f"vl_loss {vl_loss:.4f} vl_acc {vl_acc:.3f}")
    print(f"\nBest val acc: {best_val_acc:.3f}")
    model.load_state_dict(best_state)

    # 5. Evaluate
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y in test_loader:
            preds.append(model(X.to(device)).argmax(1).cpu())
            trues.append(y)
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    names = [ID2LABEL[i] for i in range(len(LABEL2ID))]
    print("\n=== Test set ===")
    print(classification_report(trues, preds, target_names=names, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(confusion_matrix(trues, preds), index=names, columns=names).to_string())

    # 6. Save checkpoint
    checkpoint = {
        "model_state_dict": best_state,
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
        "embed_dim": embed_dim,
        "plm_model_name": args.model_name,
        "best_val_acc": best_val_acc,
    }
    slug = args.model_name.replace("/", "_")
    save_path = args.models_dir / f"rbpdetect2_linear_{slug}.pt"
    torch.save(checkpoint, save_path)
    print(f"\nSaved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
