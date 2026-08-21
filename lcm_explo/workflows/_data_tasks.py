from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from prefect import get_run_logger, task

from lcm_explo.adapters.dataset.packed_tokens._class import META_FILENAME
from lcm_explo.adapters.embedding_repository.file_system import FileSystemEmbeddingRepository
from lcm_explo.domain.models.documents import DocumentEmbeddings

_SHARD_TOKENS = 50_000_000  # ~100 MB per uint16 shard
_GPT2_EOS = 50256


@task(name="encode-article-batch", persist_result=False, retries=3)
def encode_article_batch_task(
    doc_ids: list[str],
    texts: list[str],
    output_dir: Path,
    device: str = "cuda",
    min_sentence_length: int = 50,
) -> int:
    """Encode a batch of articles with SONAR and save each as a .pt file.

    SONAR (encoder + tokenizer) and the SaT splitter are loaded once for the whole batch.
    Already-encoded docs are skipped. Returns the number of docs newly encoded.
    """
    # Deferred imports: heavy modules / encode.py loads spaCy lazily.
    from transformers import NllbTokenizer  # type: ignore[import-untyped]

    from lcm_explo.adapters.splitter.sat import SaTSplitter
    from lcm_explo.constant import NLLB_TOKENIZER_HF, SONAR_ENCODER_HF
    from lcm_explo.domain.usecases.encode import encode_sentences_sonar, split_long_text
    from lcm_explo.m2m_100 import M2M100EncoderModel

    logger = get_run_logger()
    repo = FileSystemEmbeddingRepository(base_path=str(output_dir))
    existing = set(repo.list_documents())

    pending = [(doc_id, text) for doc_id, text in zip(doc_ids, texts) if doc_id not in existing]
    if not pending:
        logger.info("All %d docs already encoded — skipping batch", len(doc_ids))
        return 0

    model = M2M100EncoderModel.from_pretrained(SONAR_ENCODER_HF).to(device).eval()
    tokenizer = NllbTokenizer.from_pretrained(NLLB_TOKENIZER_HF, src_lang="eng_Latn", tgt_lang="eng_Latn")
    splitter = SaTSplitter("sat-3l")

    count = 0
    for doc_id, text in pending:
        chunks = split_long_text(splitter, text, min_sentence_length=min_sentence_length)
        embeddings = encode_sentences_sonar(chunks, device, tokenizer, model)
        if embeddings.shape[0] == 0:
            continue
        repo.save_document_embeddings(DocumentEmbeddings(document_id=doc_id, embeddings=embeddings))
        count += 1

    logger.info("Encoded %d / %d docs in batch", count, len(pending))
    return count


@task(name="fit-normalizer", persist_result=False)
def fit_normalizer_task(embeddings_dir: Path, max_samples: int = 500_000) -> Path:
    """Fit a per-dimension RobustScaler (median + IQR) on a sample of SONAR embeddings.

    Follows the Base-LCM paper (§2.3.1): normalize(x) = (x − median) / IQR.
    Uses at most `max_samples` concept vectors to keep memory bounded; the sample
    is drawn by iterating documents in order until the cap is reached.
    """
    logger = get_run_logger()
    repo = FileSystemEmbeddingRepository(base_path=str(embeddings_dir))

    chunks: list[torch.Tensor] = []
    total_seen = 0
    for doc_id in repo.list_documents():
        emb = repo.load_document_embeddings(doc_id).embeddings.float()
        if emb.shape[0] == 0:
            continue
        remaining = max_samples - total_seen
        if emb.shape[0] > remaining:
            emb = emb[:remaining]
        chunks.append(emb)
        total_seen += emb.shape[0]
        if total_seen >= max_samples:
            break

    if total_seen == 0:
        raise ValueError("No embeddings found to fit the normalizer")

    data = torch.cat(chunks, dim=0)  # (N, D), N ≤ max_samples

    # RobustScaler: median and IQR per dimension (less sensitive to outliers)
    q25 = torch.quantile(data, 0.25, dim=0)
    q75 = torch.quantile(data, 0.75, dim=0)
    median = torch.median(data, dim=0).values
    iqr = (q75 - q25).clamp(min=1e-8)

    out_path = Path(embeddings_dir) / "normalizer.pt"
    # Keys kept as "mean"/"std" so Normalizer.load_stats() works without changes.
    # Semantically: mean = median, std = IQR.
    torch.save({"mean": median, "std": iqr, "count": total_seen}, out_path)
    logger.info(
        "Fitted RobustScaler on %d concept vectors (median/IQR) → %s", total_seen, out_path
    )
    return out_path


@task(name="prepare-token-shards", persist_result=False)
def prepare_token_shards_task(
    output_dir: Path,
    seq_len: int = 1024,
    total_tokens: int = 1_000_000_000,
    sources: list[dict] | None = None,
    tokenizer_name: str = "gpt2",
    val_fraction: float = 0.01,
) -> Path:
    """Stream a web+code mix, GPT-2 tokenize, and write packed uint16 shards.

    ``sources`` is a list of ``{"path", "name", "split", "text_field", "weight",
    "filter_field", "filter_values"}`` dicts interleaved by weight. Documents are
    separated by an EOS token. ``meta.json`` is written last as a completion
    marker, so a finished directory is skipped on re-run.
    """
    from datasets import interleave_datasets, load_dataset  # type: ignore[import-untyped]
    from transformers import GPT2TokenizerFast  # type: ignore[import-untyped]

    logger = get_run_logger()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / META_FILENAME).exists():
        logger.info("Token shards already prepared at %s — skipping", output_dir)
        return output_dir

    if sources is None:
        sources = [
            {"path": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "split": "train",
             "text_field": "text", "weight": 0.75},
            {"path": "codeparrot/github-code-clean", "split": "train",
             "text_field": "code", "weight": 0.25,
             "filter_field": "language", "filter_values": ["Python", "Markdown"]},
        ]

    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else _GPT2_EOS

    streams = []
    weights = []
    for src in sources:
        ds = load_dataset(
            src["path"], src.get("name"), split=src.get("split", "train"), streaming=True
        )
        allowed = src.get("filter_values")
        if allowed is not None:
            field = src["filter_field"]
            ds = ds.filter(lambda row, f=field, a=set(allowed): row.get(f) in a)
        streams.append(ds)
        weights.append(src["weight"])

    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]
    mixed = interleave_datasets(streams, probabilities=probabilities, seed=42, stopping_strategy="all_exhausted")
    text_fields = [src["text_field"] for src in sources]

    shard_names: list[str] = []
    buffer: list[int] = []
    tokens_written = 0
    shard_idx = 0

    def flush(tokens: list[int]) -> None:
        nonlocal shard_idx
        name = f"shard_{shard_idx:04d}.bin"
        np.asarray(tokens, dtype=np.uint16).tofile(output_dir / name)
        shard_names.append(name)
        shard_idx += 1

    for row in mixed:
        text = next((row[f] for f in text_fields if row.get(f)), None)
        if not text:
            continue
        ids = tokenizer.encode(text)
        ids.append(eos)
        buffer.extend(ids)
        while len(buffer) >= _SHARD_TOKENS:
            flush(buffer[:_SHARD_TOKENS])
            buffer = buffer[_SHARD_TOKENS:]
            tokens_written += _SHARD_TOKENS
            logger.info("Wrote %d / %d tokens", tokens_written, total_tokens)
            if tokens_written >= total_tokens:
                break
        if tokens_written >= total_tokens:
            break

    if buffer and tokens_written < total_tokens:
        flush(buffer)
        tokens_written += len(buffer)

    total_sequences = max(0, (tokens_written - 1) // seq_len)
    meta = {
        "seq_len": seq_len,
        "dtype": "uint16",
        "total_tokens": tokens_written,
        "total_sequences": total_sequences,
        "val_fraction": val_fraction,
        "tokenizer": tokenizer_name,
        "sources": sources,
        "shards": shard_names,
    }
    (output_dir / META_FILENAME).write_text(json.dumps(meta, indent=2))
    logger.info("Prepared %d tokens (%d sequences) → %s", tokens_written, total_sequences, output_dir)
    return output_dir
