from __future__ import annotations

import torch


class SonarTextDecoder:
    """Decode SONAR sentence embeddings (1024-d) back to text.

    Uses the HuggingFace-native port ``raxtemur/SONAR_200_text_decoder`` (no fairseq2).
    The embedding is injected as the decoder's cross-attention memory and text is
    produced with the standard ``generate`` loop.

    Note: this wraps a community port. If the checkpoint's interface differs, adjust the
    loading / ``encoder_outputs`` plumbing against the model card — the rest of the
    pipeline does not depend on this adapter.
    """

    def __init__(self, device: str = "cpu") -> None:
        from transformers import NllbTokenizer  # type: ignore[import-untyped]
        from transformers.modeling_outputs import BaseModelOutput

        from lcm_explo.constant import NLLB_TOKENIZER_HF, SONAR_DECODER_HF
        from lcm_explo.m2m_100 import M2M100ForConditionalGeneration

        self._device = device
        self._base_model_output_cls = BaseModelOutput
        self._model = M2M100ForConditionalGeneration.from_pretrained(SONAR_DECODER_HF).to(device).eval()
        self._tokenizer = NllbTokenizer.from_pretrained(
            NLLB_TOKENIZER_HF, src_lang="eng_Latn", tgt_lang="eng_Latn"
        )

    @torch.no_grad()
    def decode(self, embeddings: torch.Tensor, target_lang: str = "eng_Latn", max_length: int = 256) -> list[str]:
        """embeddings: (N, 1024) SONAR vectors → list of N decoded strings."""
        embeddings = embeddings.to(self._device)
        encoder_outputs = self._base_model_output_cls(last_hidden_state=embeddings.unsqueeze(1))
        generated = self._model.generate(
            encoder_outputs=encoder_outputs,
            forced_bos_token_id=self._tokenizer.lang_code_to_id[target_lang],
            max_length=max_length,
        )
        return self._tokenizer.batch_decode(generated, skip_special_tokens=True)
