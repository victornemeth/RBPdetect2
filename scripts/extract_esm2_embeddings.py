#!/usr/bin/env python3
"""Extract frozen ESM2-650M sequence embeddings for the benchmark dataset."""

import argparse

from rbpdetect2.embedding_cli import add_common_arguments, run_hf_extractor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(
        parser,
        default_model_id="facebook/esm2_t33_650M_UR50D",
        default_output_dir="benchmark_embeddings/embeddings/esm2-650m",
    )
    args = parser.parse_args()
    run_hf_extractor(args, model_key="esm2-650m")


if __name__ == "__main__":
    main()

