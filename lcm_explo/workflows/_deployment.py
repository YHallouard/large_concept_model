"""
Deploy training and data-preparation flows to the gpu-homelab Prefect work pool.

Usage:
    uv run python -m lcm_explo.workflows._deployment

Prerequisites:
    1. Create the work pool in Prefect UI or CLI:
       prefect work-pool create gpu-homelab --type process

    2. Start the worker on the GPU machine:
       PREFECT_API_URL=http://<prefect-server>:4200/api prefect worker start --pool gpu-homelab

    3. Run this script once to register the deployments.
"""

from pathlib import Path

from lcm_explo.domain.models.one_tower_lcm import OneTowerLCMConfig
from lcm_explo.utils._settings import settings
from lcm_explo.utils.checkpoints import S3CheckpointStorage
from lcm_explo.workflows._data_flows import PrepareDataConfig, prepare_lcm_data_flow
from lcm_explo.workflows._flows import train_base_lcm_flow, train_one_tower_lcm_flow
from lcm_explo.workflows._inputs import (
    DataConfig,
    PresetModelSpec,
    TrainBaseLCMConfig,
    TrainOneTowerLCMConfig,
    TrainingConfig,
)

WORK_POOL = "gpu-homelab"
EMBEDDINGS_DIR = Path("/data/embeddings/wikipedia")  # adjust to actual path on GPU machine

default_train_config = TrainBaseLCMConfig(
    data=DataConfig(embeddings_dir=EMBEDDINGS_DIR),
    model=PresetModelSpec(size="base"),
    training=TrainingConfig(
        learning_rate=2e-4,
        warmup_steps=2000,
        max_steps=100_000,
        batch_size=8,
        checkpoint_every_n_steps=500,
    ),
    checkpoint_storage=S3CheckpointStorage(bucket=settings.MINIO_BUCKET),
    mlflow_tracking_uri=settings.MLFLOW_TRACKING_URI,
    experiment_name="base-lcm",
)

default_one_tower_config = TrainOneTowerLCMConfig(
    data=DataConfig(embeddings_dir=EMBEDDINGS_DIR),
    model=OneTowerLCMConfig(num_hidden_layers=12),
    training=TrainingConfig(
        learning_rate=2e-4,
        warmup_steps=2000,
        max_steps=100_000,
        batch_size=8,
        checkpoint_every_n_steps=500,
    ),
    checkpoint_storage=S3CheckpointStorage(bucket=settings.MINIO_BUCKET),
    mlflow_tracking_uri=settings.MLFLOW_TRACKING_URI,
    experiment_name="one-tower-lcm",
)

default_data_config = PrepareDataConfig(
    dataset_name="wikipedia",
    dataset_language="en",
    output_dir=EMBEDDINGS_DIR,
    device="cuda",
    batch_size=1000,
)

if __name__ == "__main__":
    prepare_lcm_data_flow.deploy(
        name="prepare-lcm-data-gpu",
        work_pool_name=WORK_POOL,
        parameters={"config": default_data_config.model_dump()},
    )
    print(f"Deployed 'prepare-lcm-data-gpu' → work pool '{WORK_POOL}'")

    train_base_lcm_flow.deploy(
        name="train-base-lcm-gpu",
        work_pool_name=WORK_POOL,
        parameters={"config": default_train_config.model_dump()},
    )
    print(f"Deployed 'train-base-lcm-gpu' → work pool '{WORK_POOL}'")

    train_one_tower_lcm_flow.deploy(
        name="train-one-tower-lcm-gpu",
        work_pool_name=WORK_POOL,
        parameters={"config": default_one_tower_config.model_dump()},
    )
    print(f"Deployed 'train-one-tower-lcm-gpu' → work pool '{WORK_POOL}'")
