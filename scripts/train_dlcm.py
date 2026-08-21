#!/usr/bin/env python3
"""CLI to train a Dynamic Large Concept Model (DLCM).

Example (small model on 1 GPU):
    uv run python scripts/train_dlcm.py \
        --tokens-dir notebooks/data/dlcm_tokens \
        --output-dir notebooks/data/checkpoints \
        --max-steps 30_000 \
        --micro-batch-size 4 \
        --accumulate-grad-batches 8

The script is a thin wrapper around ``train_dlcm_flow``; it uses the same
Pydantic configs and Lightning/Prefect plumbing as the workflow API.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lcm_explo.workflows._flows import TrainDLCMConfig, train_dlcm_flow
from lcm_explo.workflows._inputs import DLCMTrainingConfig, PresetDLCMModelSpec, TokenDataConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Dynamic Large Concept Model")
    parser.add_argument("--tokens-dir", type=Path, required=True, help="Directory containing packed token shards")
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Directory for checkpoints/logs")
    parser.add_argument("--size", type=str, default="small", choices=["tiny", "small"], help="Model preset")
    parser.add_argument("--max-steps", type=int, default=100_000, help="Training steps")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--warmup-steps", type=int, default=2_000, help="Linear warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="AdamW weight decay")
    parser.add_argument("--micro-batch-size", type=int, default=4, help="Per-device micro-batch")
    parser.add_argument("--accumulate-grad-batches", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--aux-loss-weight", type=float, default=0.03, help="Weight of boundary-ratio loss")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Mixed precision")
    parser.add_argument("--val-check-interval", type=int, default=1_000, help="Validation every N steps")
    parser.add_argument("--checkpoint-every-n-steps", type=int, default=500, help="Checkpoint frequency")
    parser.add_argument("--warm-start-embedding", action="store_true", help="Warm-start GPT-2 embedding via PCA")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--experiment-name", type=str, default="dlcm", help="MLflow experiment name")
    args = parser.parse_args()

    config = TrainDLCMConfig(
        model=PresetDLCMModelSpec(size=args.size),  # type: ignore[arg-type]
        data=TokenDataConfig(tokens_dir=args.tokens_dir, num_workers=args.num_workers),
        training=DLCMTrainingConfig(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            aux_loss_weight=args.aux_loss_weight,
            micro_batch_size=args.micro_batch_size,
            accumulate_grad_batches=args.accumulate_grad_batches,
            precision=args.precision,
            val_check_interval=args.val_check_interval,
            checkpoint_every_n_steps=args.checkpoint_every_n_steps,
            warm_start_embedding=args.warm_start_embedding,
            seed=args.seed,
        ),
        experiment_name=args.experiment_name,
    )
    result = train_dlcm_flow(config)
    print(f"Run {result.run_id} finished at step {result.final_step}. Best val loss: {result.best_val_loss}")


if __name__ == "__main__":
    main()
