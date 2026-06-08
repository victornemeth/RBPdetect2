"""Shared frozen-PLM embedding helpers for training and inference.

Mean-pools the last hidden state over residue tokens (excluding BOS/EOS/PAD),
matching the pooling used in notebooks/train_classifier.ipynb. Sequences longer
than ``max_residues`` are split into non-overlapping chunks and combined with a
residue-weighted mean, so embeddings stay consistent regardless of length.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


def select_device(requested: str = "auto") -> torch.device:
    """Pick the requested device, or the CUDA GPU with the most free memory."""
    if requested != "auto":
        return torch.device(requested)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    return torch.device(f"cuda:{int(np.argmax(free))}")


def select_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[requested]


def load_plm(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.nn.Module, Any, int]:
    """Load a frozen masked-LM PLM and its tokenizer. Returns (model, tokenizer, embed_dim)."""
    model = (
        AutoModelForMaskedLM.from_pretrained(model_name, dtype=dtype)
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer, int(model.config.hidden_size)


def _chunks(sequences: list[str], max_residues: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record_index, seq in enumerate(sequences):
        for start in range(0, max(len(seq), 1), max_residues):
            end = min(start + max_residues, len(seq))
            out.append({"record_index": record_index, "text": seq[start:end]})
    return out


@torch.inference_mode()
def embed_sequences(
    model: torch.nn.Module,
    tokenizer: Any,
    sequences: list[str],
    device: torch.device,
    *,
    batch_size: int = 8,
    max_residues: int = 1022,
    show_progress: bool = True,
) -> np.ndarray:
    """Mean-pooled last-hidden-state embeddings, shape (len(sequences), embed_dim)."""
    chunks = _chunks(sequences, max_residues)
    embed_dim = int(model.config.hidden_size)
    sums = np.zeros((len(sequences), embed_dim), dtype=np.float64)
    counts = np.zeros(len(sequences), dtype=np.int64)

    iterator = range(0, len(chunks), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Embedding")

    for offset in iterator:
        batch = chunks[offset : offset + batch_size]
        encoded = tokenizer(
            [c["text"] for c in batch],
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_special_tokens_mask=True,
        )
        special = encoded.pop("special_tokens_mask").bool()
        attention = encoded["attention_mask"].bool()
        inputs = {k: v.to(device) for k, v in encoded.items()}

        hidden = model(**inputs, output_hidden_states=True, return_dict=True)
        hidden = hidden.hidden_states[-1].float().cpu()
        residue_mask = attention & ~special

        for i, chunk in enumerate(batch):
            n = int(residue_mask[i].sum())
            if n == 0:  # empty sequence — skip, leaves a zero vector
                continue
            vec = hidden[i][residue_mask[i]].mean(dim=0).numpy().astype(np.float64)
            sums[chunk["record_index"]] += vec * n
            counts[chunk["record_index"]] += n

    counts = np.where(counts == 0, 1, counts)
    return (sums / counts[:, None]).astype(np.float32)
