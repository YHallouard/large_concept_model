from __future__ import annotations

from pathlib import Path
from typing import Literal

from prefect import flow, get_run_logger
from pydantic import BaseModel

from lcm_explo.workflows._data_tasks import (
    encode_article_batch_task,
    fit_normalizer_task,
    prepare_token_shards_task,
)


class PrepareDataConfig(BaseModel):
    dataset_name: Literal["wikipedia", "logicot", "roc_stories"] = "wikipedia"
    dataset_language: str = "en"
    output_dir: Path
    device: str = "cuda"
    max_articles: int | None = None
    min_sentence_length: int = 50
    batch_size: int = 1000  # articles per Prefect task


class PrepareDataResult(BaseModel):
    num_documents_encoded: int
    num_documents_skipped: int
    output_dir: Path
    normalizer_path: Path


def _iter_dataset(
    dataset_name: str,
    language: str,
    max_articles: int | None,
) -> list[tuple[str, str]]:
    """Return (doc_id, text) pairs streamed from a HuggingFace dataset."""
    from datasets import load_dataset  # type: ignore[import-untyped]

    if dataset_name == "wikipedia":
        dataset = load_dataset("wikimedia/wikipedia", f"20231101.{language}", split="train", streaming=True)
    elif dataset_name == "logicot":
        dataset = load_dataset("datatune/LogiCoT", split="train", streaming=True)
    elif dataset_name == "roc_stories":
        dataset = load_dataset("mintujupally/ROCStories", split="train", streaming=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    pairs: list[tuple[str, str]] = []
    for i, row in enumerate(dataset):
        if max_articles is not None and i >= max_articles:
            break
        if dataset_name == "wikipedia":
            doc_id = str(row.get("id", i))
            text = row.get("text") or ""
        elif dataset_name == "logicot":
            doc_id = f"logicot_entry_{i}"
            modified_input = (row.get("input") or "").replace("sent", ". sent")
            text = f"{row.get('instruction', '')} {modified_input} {row.get('output', '')}"
        else:  # roc_stories
            doc_id = f"roc_story_{i}"
            text = row.get("text") or row.get("story") or ""
        if text.strip():
            pairs.append((doc_id, text))
    return pairs


class PrepareDlcmTokensConfig(BaseModel):
    output_dir: Path
    seq_len: int = 1024
    total_tokens: int = 1_000_000_000
    tokenizer_name: str = "gpt2"
    val_fraction: float = 0.01
    sources: list[dict] | None = None


class PrepareDlcmTokensResult(BaseModel):
    tokens_dir: Path


@flow(name="prepare-dlcm-tokens", log_prints=True)
def prepare_dlcm_tokens_flow(config: PrepareDlcmTokensConfig) -> PrepareDlcmTokensResult:
    logger = get_run_logger()
    logger.info("Preparing %d packed tokens → %s", config.total_tokens, config.output_dir)
    tokens_dir = prepare_token_shards_task(
        output_dir=config.output_dir,
        seq_len=config.seq_len,
        total_tokens=config.total_tokens,
        sources=config.sources,
        tokenizer_name=config.tokenizer_name,
        val_fraction=config.val_fraction,
    )
    return PrepareDlcmTokensResult(tokens_dir=tokens_dir)


@flow(name="prepare-lcm-data", log_prints=True)
def prepare_lcm_data_flow(config: PrepareDataConfig) -> PrepareDataResult:
    logger = get_run_logger()
    logger.info(
        "Loading %s/%s (max=%s)",
        config.dataset_name,
        config.dataset_language,
        config.max_articles or "all",
    )

    pairs = _iter_dataset(config.dataset_name, config.dataset_language, config.max_articles)
    logger.info("Loaded %d articles", len(pairs))

    batches = [pairs[i : i + config.batch_size] for i in range(0, len(pairs), config.batch_size)]
    logger.info("Submitting %d batches of ~%d articles", len(batches), config.batch_size)

    futures = [
        encode_article_batch_task.submit(
            doc_ids=[doc_id for doc_id, _ in batch],
            texts=[text for _, text in batch],
            output_dir=config.output_dir,
            device=config.device,
            min_sentence_length=config.min_sentence_length,
        )
        for batch in batches
    ]

    total_encoded = sum(f.result() for f in futures)
    total_skipped = len(pairs) - total_encoded

    normalizer_path = fit_normalizer_task(config.output_dir)

    logger.info("Done — encoded %d, skipped %d", total_encoded, total_skipped)
    return PrepareDataResult(
        num_documents_encoded=total_encoded,
        num_documents_skipped=total_skipped,
        output_dir=config.output_dir,
        normalizer_path=normalizer_path,
    )
