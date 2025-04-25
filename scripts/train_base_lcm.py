#!/usr/bin/env python3
import argparse
from pathlib import Path

import lightning as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

from lcm_explo.adapters.dataset.embedding import EmbeddingsDataset
from lcm_explo.adapters.dataset.embedding._class import embedding_collate_fn

# from lcm_explo.adapters.dataset.random_sized_embeddings import RandomSequenceDataset
from lcm_explo.domain.models.base_lcm import BaseLCMConfig
from lcm_explo.domain.usecases.train import BaseLCMTrainingModule


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the LCM model")
    parser.add_argument("--embeddings-dir", type=Path, required=True, help="Directory containing the embeddings")
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Directory to save model checkpoints")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--max-epochs", type=int, default=30, help="Maximum number of epochs to train")
    parser.add_argument("--sequence-length", type=int, default=32, help="Length of sequences for training")
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset and dataloaders
    dataset = EmbeddingsDataset(embeddings_dir=args.embeddings_dir, sequence_length=args.sequence_length)
    # dataset = RandomSequenceDataset(
    #     embeddings_dir=args.embeddings_dir,
    #     min_length=2,
    #     max_length=args.sequence_length
    # )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

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
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        log_every_n_steps=1,
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
