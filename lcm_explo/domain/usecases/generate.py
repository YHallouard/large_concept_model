"""Qualitative generation and segmentation inspection for the DLCM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from lcm_explo.domain.models.dlcm import DLCM


@dataclass
class SegmentSpan:
    """A discovered concept: the token span it covers and its decoded text."""

    start: int
    end: int  # exclusive
    text: str


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int]
    segments: list[SegmentSpan]


def _spans_from_boundaries(boundaries: torch.Tensor) -> list[tuple[int, int]]:
    """Turn a 1-D boundary indicator into (start, end) token spans."""
    starts = [i for i, b in enumerate(boundaries.tolist()) if b > 0]
    if not starts:
        return [(0, boundaries.numel())]
    ends = starts[1:] + [boundaries.numel()]
    return list(zip(starts, ends))


def generate_dlcm(
    model: DLCM,
    tokenizer: object,
    prompt: str,
    max_new_tokens: int = 128,
    top_k: int | None = 50,
    temperature: float = 1.0,
) -> GenerationResult:
    """Generate a continuation and report the concepts the model discovered.

    ``tokenizer`` is any HF tokenizer exposing ``encode`` / ``decode``. The
    returned segment spans expose the learned dynamic segmentation over the full
    generated sequence, for eyeballing whether boundaries align with word or
    phrase units.
    """
    device = next(model.parameters()).device
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)  # type: ignore[attr-defined]

    generated = model.generate(
        input_ids, max_new_tokens=max_new_tokens, top_k=top_k, temperature=temperature
    )
    token_ids = generated[0].tolist()

    boundaries = model.segment(generated[:, : model.config.max_seq_len])[0]
    spans = _spans_from_boundaries(boundaries)
    segments = [
        SegmentSpan(start=s, end=e, text=tokenizer.decode(token_ids[s:e]))  # type: ignore[attr-defined]
        for s, e in spans
    ]

    return GenerationResult(
        text=tokenizer.decode(token_ids),  # type: ignore[attr-defined]
        token_ids=token_ids,
        segments=segments,
    )
