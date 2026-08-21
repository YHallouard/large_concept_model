"""Warm-start the DLCM token embedding from a pretrained GPT-2 embedding table.

Speeds up convergence on a limited token budget: instead of a random init, the
embedding starts from GPT-2's learned token statistics, dimensionality-reduced
to the model's `d_token` via PCA. This only re-initializes weights — no frozen
external model is kept around at train/inference time.
"""

from __future__ import annotations

import logging

import torch

from lcm_explo.domain.models.dlcm import DLCM

logger = logging.getLogger(__name__)


def _project_embedding_pca(source_weight: torch.Tensor, d_token: int, target_std: float = 0.02) -> torch.Tensor:
    """Reduce a (vocab, source_dim) embedding table to (vocab, d_token) via PCA.

    Keeps the top-`d_token` principal components (the directions of highest
    variance in the source embedding space) via ``torch.pca_lowrank``, then
    rescales so the result's std matches ``target_std`` — keeping the residual
    stream at the same order of magnitude the model's random init would have
    produced (DLCMConfig.initializer_range), so training stays stable.
    """
    source_vocab, source_dim = source_weight.shape
    if d_token > source_dim:
        raise ValueError(
            f"d_token={d_token} exceeds the source embedding dimension={source_dim}; "
            "cannot warm-start by dimensionality reduction."
        )

    # A ≈ U diag(S) Vᵀ (centered) ⇒ A_centered @ V ≈ U·S : the projection onto
    # the top-d_token principal components, without recomputing the product.
    u, s, _ = torch.pca_lowrank(source_weight.float(), q=d_token, center=True)
    projected = u * s
    return projected / projected.std() * target_std


def warm_start_embedding_from_gpt2(model: DLCM, source_model_name: str = "gpt2") -> bool:
    """Warm-start ``model``'s token embedding from a pretrained GPT-2 table.

    Only applies when ``model``'s vocabulary matches the source's (token ids
    must correspond 1:1) — otherwise this is a no-op that logs a warning and
    returns False. Mutates the embedding in place; if the LM head is tied it
    updates automatically (shared storage), otherwise it is warm-started with
    the same projected table.
    """
    from transformers import GPT2Model  # deferred: heavy, network-fetching import

    source = GPT2Model.from_pretrained(source_model_name)
    source_weight = source.wte.weight.data

    vocab_size = model.config.vocab_size
    if vocab_size != source_weight.shape[0]:
        logger.warning(
            "Skipping embedding warm-start: model vocab_size=%d does not match "
            "%s vocab_size=%d (token ids would not correspond).",
            vocab_size,
            source_model_name,
            source_weight.shape[0],
        )
        return False

    projected = _project_embedding_pca(source_weight, model.config.d_token, model.config.initializer_range)

    with torch.no_grad():
        model.encoder.embedding.weight.copy_(projected.to(model.encoder.embedding.weight.dtype))
        if not model.config.tie_embeddings:
            model.lm_head.weight.copy_(projected.to(model.lm_head.weight.dtype))

    logger.info("Warm-started embedding (d_token=%d) from %s via PCA.", model.config.d_token, source_model_name)
    return True
