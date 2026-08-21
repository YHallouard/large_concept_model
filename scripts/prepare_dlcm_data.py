#!/usr/bin/env python3
"""CLI to prepare packed token shards for DLCM pre-training.

Example:
    uv run python scripts/prepare_dlcm_data.py \
        --output-dir notebooks/data/dlcm_tokens \
        --total-tokens 1_000_000_000 \
        --seq-len 1024
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lcm_explo.workflows._data_flows import PrepareDlcmTokensConfig, prepare_dlcm_tokens_flow


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare packed GPT-2 token shards for DLCM training")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write token shards")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length (default: 1024)")
    parser.add_argument(
        "--total-tokens",
        type=int,
        default=1_000_000_000,
        help="Total number of tokens to stream (default: 1B)",
    )
    parser.add_argument("--tokenizer-name", type=str, default="gpt2", help="Tokenizer to use (default: gpt2)")
    parser.add_argument("--val-fraction", type=float, default=0.01, help="Validation split fraction (default: 0.01)")
    args = parser.parse_args()

    config = PrepareDlcmTokensConfig(
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        total_tokens=args.total_tokens,
        tokenizer_name=args.tokenizer_name,
        val_fraction=args.val_fraction,
    )
    prepare_dlcm_tokens_flow(config)


if __name__ == "__main__":
    main()
