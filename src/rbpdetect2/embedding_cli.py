"""Hugging Face embedding extraction used by the model-specific CLI scripts."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from rbpdetect2.benchmark_data import (
    dataset_sha256,
    load_labeled_sequences,
    sequence_sha256,
)


TextBuilder = Callable[[dict[str, str], int, int], str]


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_model_id: str,
    default_output_dir: str,
) -> None:
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path(default_output_dir))
    parser.add_argument("--model-id", default=default_model_id)
    parser.add_argument(
        "--revision",
        default=None,
        help="Hugging Face model revision, preferably a commit hash for reproducibility",
    )
    parser.add_argument(
        "--max-residues",
        type=int,
        default=1022,
        help="Maximum residues per model call; longer proteins are chunked and pooled",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Number of chunks per model call")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a device such as cuda:0")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")


def run_hf_extractor(
    args: argparse.Namespace,
    *,
    model_key: str,
    text_builder: TextBuilder | None = None,
    input_kind: str = "amino_acid_sequence",
    low_cpu_mem_usage: bool | None = None,
    records: list[dict[str, str]] | None = None,
) -> None:
    if args.max_residues < 1:
        raise ValueError("--max-residues must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    if records is None:
        records = load_labeled_sequences(args.data_dir)
    if text_builder is None:
        def amino_acid_text_builder(record: dict[str, str], start: int, end: int) -> str:
            return record["sequence"][start:end]

        text_builder = amino_acid_text_builder

    output_paths = _output_paths(args.output_dir)
    _prepare_output(output_paths, overwrite=args.overwrite)

    device = _select_device(args.device)
    dtype = _select_dtype(args.dtype, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )
    model_kwargs: dict[str, Any] = {
        "revision": args.revision,
        "trust_remote_code": args.trust_remote_code,
        "dtype": dtype,
    }
    if low_cpu_mem_usage is not None:
        model_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage

    started = time.perf_counter()
    model = AutoModelForMaskedLM.from_pretrained(args.model_id, **model_kwargs).to(device).eval()
    chunks = _build_chunks(records, args.max_residues, text_builder)
    embeddings = _embed_chunks(
        model=model,
        tokenizer=tokenizer,
        chunks=chunks,
        n_records=len(records),
        batch_size=args.batch_size,
        device=device,
    )
    elapsed_seconds = time.perf_counter() - started

    np.save(output_paths["embeddings"], embeddings, allow_pickle=False)
    _write_ids(output_paths["ids"], records)
    metadata = _metadata(
        args=args,
        records=records,
        embeddings=embeddings,
        model=model,
        model_key=model_key,
        input_kind=input_kind,
        device=device,
        dtype=dtype,
        elapsed_seconds=elapsed_seconds,
    )
    output_paths["metadata"].write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(f"Saved {embeddings.shape} embeddings to {args.output_dir}")
    print(f"Dataset SHA256: {metadata['dataset_sha256']}")
    print(f"Elapsed: {elapsed_seconds:.1f} seconds")


def _build_chunks(
    records: list[dict[str, str]],
    max_residues: int,
    text_builder: TextBuilder,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        sequence_length = len(record["sequence"])
        for start in range(0, sequence_length, max_residues):
            end = min(start + max_residues, sequence_length)
            chunks.append(
                {
                    "record_index": record_index,
                    "id": record["id"],
                    "text": text_builder(record, start, end),
                    "n_residues": end - start,
                }
            )
    return chunks


@torch.inference_mode()
def _embed_chunks(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    chunks: list[dict[str, Any]],
    n_records: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    sums: list[np.ndarray | None] = [None] * n_records
    residue_counts = np.zeros(n_records, dtype=np.int64)

    for offset in tqdm(range(0, len(chunks), batch_size), desc="Embedding chunks"):
        batch = chunks[offset : offset + batch_size]
        encoded = tokenizer(
            [chunk["text"] for chunk in batch],
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_special_tokens_mask=True,
        )
        special_tokens_mask = encoded.pop("special_tokens_mask").bool()
        attention_mask = encoded["attention_mask"].bool()
        model_inputs = {key: value.to(device) for key, value in encoded.items()}

        output = model(**model_inputs, output_hidden_states=True, return_dict=True)
        hidden = output.hidden_states[-1].float().cpu()
        residue_mask = attention_mask & ~special_tokens_mask

        for batch_index, chunk in enumerate(batch):
            token_count = int(residue_mask[batch_index].sum())
            if token_count != chunk["n_residues"]:
                raise ValueError(
                    f"Tokenizer produced {token_count} residue tokens for {chunk['id']} chunk, "
                    f"but {chunk['n_residues']} residues were expected"
                )
            chunk_embedding = hidden[batch_index][residue_mask[batch_index]].mean(dim=0).numpy()
            record_index = chunk["record_index"]
            weighted = chunk_embedding.astype(np.float64) * chunk["n_residues"]
            if sums[record_index] is None:
                sums[record_index] = weighted
            else:
                sums[record_index] += weighted
            residue_counts[record_index] += chunk["n_residues"]

    if any(value is None for value in sums):
        raise RuntimeError("At least one sequence did not receive an embedding")
    return np.stack(
        [value / residue_counts[index] for index, value in enumerate(sums)]
    ).astype(np.float32)


def _select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    free_memory = [torch.cuda.mem_get_info(index)[0] for index in range(torch.cuda.device_count())]
    return torch.device(f"cuda:{int(np.argmax(free_memory))}")


def _select_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[requested]


def _output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "dir": output_dir,
        "embeddings": output_dir / "embeddings.npy",
        "ids": output_dir / "ids.tsv",
        "metadata": output_dir / "metadata.json",
    }


def _prepare_output(paths: dict[str, Path], *, overwrite: bool) -> None:
    paths["dir"].mkdir(parents=True, exist_ok=True)
    existing = [path for key, path in paths.items() if key != "dir" and path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists: {names}. Use --overwrite to replace it.")


def _write_ids(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_index", "id", "label", "sequence_length", "sequence_sha256"],
            delimiter="\t",
        )
        writer.writeheader()
        for index, record in enumerate(records):
            writer.writerow(
                {
                    "row_index": index,
                    "id": record["id"],
                    "label": record["label"],
                    "sequence_length": len(record["sequence"]),
                    "sequence_sha256": sequence_sha256(record["sequence"]),
                }
            )


def _metadata(
    *,
    args: argparse.Namespace,
    records: list[dict[str, str]],
    embeddings: np.ndarray,
    model: torch.nn.Module,
    model_key: str,
    input_kind: str,
    device: torch.device,
    dtype: torch.dtype,
    elapsed_seconds: float,
) -> dict[str, Any]:
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    return {
        "schema_version": 1,
        "model_key": model_key,
        "model_id": args.model_id,
        "requested_revision": args.revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "input_kind": input_kind,
        "pooling": "last_hidden_state_mean_over_residues",
        "long_sequence_policy": "non_overlapping_chunks_then_residue_weighted_mean",
        "max_residues_per_chunk": args.max_residues,
        "n_sequences": len(records),
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_dtype": str(embeddings.dtype),
        "dataset_sha256": dataset_sha256(records),
        "elapsed_seconds": elapsed_seconds,
        "peak_gpu_memory_bytes": peak_memory,
        "device": str(device),
        "gpu_name": gpu_name,
        "model_dtype": str(dtype),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }

