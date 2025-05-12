#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

from lcm_explo.adapters.dataset.embedding import EmbeddingsDataset
from lcm_explo.adapters.dataset.embedding._class import embedding_collate_fn

# from lcm_explo.adapters.dataset.random_sized_embeddings import RandomSequenceDataset
from lcm_explo.domain.models.base_lcm import BaseLCMConfig
from lcm_explo.domain.usecases.train import BaseLCMTrainingModule


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the LCM model")
    parser.add_argument("--embeddings-dir", type=Path, required=True, help="Directory containing the embeddings")
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Directory to save model checkpoints")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--max-epochs", type=int, default=30, help="Maximum number of epochs to train")
    parser.add_argument("--sequence-length", type=int, default=32, help="Length of sequences for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--resume-from-checkpoint", type=Path, default=None, help="Path to checkpoint for resuming training"
    )
    args = parser.parse_args()

    # Set random seed
    set_random_seed(args.seed)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset and dataloaders
    dataset = EmbeddingsDataset(embeddings_dir=args.embeddings_dir, sequence_length=args.sequence_length)
    # dataset = RandomSequenceDataset(
    #     embeddings_dir=args.embeddings_dir,
    #     min_length=2,
    #     max_length=args.sequence_length
    # )

    # Split documents into train and validation sets
    train_docs = int(0.9 * len(dataset.documents))
    train_documents = dataset.documents[:train_docs]
    val_documents = dataset.documents[train_docs:]

    train_sequence_indices = [idx for idx in dataset.sequence_indices if idx[0] in train_documents]
    val_sequence_indices = [idx for idx in dataset.sequence_indices if idx[0] in val_documents]

    train_dataset = torch.utils.data.Subset(
        dataset, [dataset.sequence_indices.index(idx) for idx in train_sequence_indices]
    )
    val_dataset = torch.utils.data.Subset(
        dataset, [dataset.sequence_indices.index(idx) for idx in val_sequence_indices]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=embedding_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=embedding_collate_fn,
    )

    # Create model and trainer
    config = BaseLCMConfig(
        num_hidden_layers=32,
        max_seq_len=args.sequence_length,
    )
    model = BaseLCMTrainingModule(config)

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.output_dir,
            filename="lcm-{epoch:02d}-{val_loss:.4f}",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="mps",
        callbacks=callbacks,
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
        log_every_n_steps=1,
        precision="16-mixed",
        enable_model_summary=True,
        enable_progress_bar=True,
        enable_checkpointing=True,
        deterministic=False,
        benchmark=True,
        profiler="simple",
    )

    # Train model
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
