#!/usr/bin/env python3
"""Extract frozen ESMC-6B sequence embeddings for the benchmark dataset."""

import argparse

from rbpdetect2.embedding_cli import add_common_arguments, run_hf_extractor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(
        parser,
        default_model_id="biohub/ESMC-6B",
        default_output_dir="benchmark_embeddings/embeddings/esmc-6b",
    )
    args = parser.parse_args()
    run_hf_extractor(
        args,
        model_key="esmc-6b",
        low_cpu_mem_usage=False,
    )


if __name__ == "__main__":
    main()

