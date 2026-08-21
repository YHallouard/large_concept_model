import torch

from lcm_explo.domain.infrastructures.tokenizer import Tokenizer


class InMemoryTokenizer(Tokenizer):
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size

    def __call__(
        self,
        text: str | list[str],
        return_tensors: str = "pt",
        padding: bool = False,
        truncation: bool = False,
    ) -> dict[str, torch.Tensor]:
        texts = [text] if isinstance(text, str) else text
        token_id_lists = [[hash(token) % self.vocab_size for token in t.split(" ")] for t in texts]
        max_len = max((len(ids) for ids in token_id_lists), default=0)

        input_ids = []
        attention_mask = []
        for ids in token_id_lists:
            pad = max_len - len(ids)
            input_ids.append(ids + [0] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }
