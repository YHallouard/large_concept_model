from functools import lru_cache
from typing import Any

import torch

from lcm_explo.constant import SONAR_DIMENSIONS
from lcm_explo.domain.infrastructures.splitter import TextSplitter
from lcm_explo.domain.infrastructures.tokenizer import Tokenizer


@lru_cache(maxsize=1)
def _get_nlp() -> Any:
    """Lazily load the spaCy model. Deferred so importing this module has no side effect."""
    import spacy

    return spacy.load("en_core_web_sm")


def split_long_text(splitter: TextSplitter, text: str, min_sentence_length: int = 50) -> list[str]:
    splits = splitter.split(text)
    splits = [
        split.removesuffix(" ") for split in splits if split.replace(" ", "") != "" and len(split) > min_sentence_length
    ]
    return splits


def encode_sentences_sonar(
    sentences: list[str],
    device: str,
    sonar_tokenizer: Tokenizer,
    sonar_model: torch.nn.Module,
    batch_size: int = 32,
) -> torch.Tensor:
    """Encode a flat list of sentences into SONAR embeddings.

    Sentences are tokenized in padded batches and run through the encoder in a single
    forward pass each; the sentence vector is a masked mean-pool over non-padding tokens.

    Returns:
        Tensor of shape (len(sentences), SONAR_DIMENSIONS) on CPU.
    """
    if not sentences:
        return torch.zeros((0, SONAR_DIMENSIONS))

    embeddings = []
    with torch.no_grad():
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start : start + batch_size]
            inputs = sonar_tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = sonar_model(**inputs)
            hidden = outputs.last_hidden_state  # (B, T, D)
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # (B, T, 1)
            summed = (hidden * mask).sum(dim=1)  # (B, D)
            counts = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
            embeddings.append((summed / counts).cpu())
    return torch.vstack(embeddings)


def encode_text_sonar(
    text: str,
    device: str,
    sonar_tokenizer: Tokenizer,
    sonar_model: torch.nn.Module,
    splitter: TextSplitter,
    min_sentence_length: int = 50,
) -> torch.Tensor:
    chunks = split_long_text(splitter, text, min_sentence_length=min_sentence_length)
    return encode_sentences_sonar(chunks, device, sonar_tokenizer, sonar_model)


def encode_text_sonar_with_spacy(
    text: str,
    device: str,
    sonar_tokenizer: Tokenizer,
    sonar_model: torch.nn.Module,
    max_sentence_length: int = 75,
) -> torch.Tensor:
    chunks = split_text(text, max_sentence_length)
    return encode_sentences_sonar(chunks, device, sonar_tokenizer, sonar_model)


def spacy_segment(text: str) -> list[str]:
    doc = _get_nlp()(text)
    return [sent.text.strip().replace("\n", " ") for sent in doc.sents]


def fallback_resplit(text: str, max_length: int) -> list[str]:
    words = text.split()
    result = []
    current_piece = ""
    for word in words:
        if len(current_piece) + len(word) + 1 <= max_length:
            current_piece += word + " "
        else:
            result.append(current_piece.strip())
            current_piece = word + " "
    if current_piece:
        result.append(current_piece.strip())
    return result


def split_text(text: str, max_length: int) -> list[str]:
    sentences = []
    for sent in spacy_segment(text):
        if len(sent) < max_length:
            sentences.append(sent)
        else:
            sentences.extend(fallback_resplit(sent, max_length))
    return sentences
