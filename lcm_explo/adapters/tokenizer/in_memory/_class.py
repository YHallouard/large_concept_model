import torch

from lcm_explo.domain.infrastructures.tokenizer import Tokenizer


class InMemoryTokenizer(Tokenizer):
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size

    def __call__(
        self, text: str, return_tensors: str = "pt", padding: bool = False, truncation: bool = False
    ) -> dict[str, torch.Tensor]:
        tokens = text.split(" ")
        token_ids = [hash(token) % self.vocab_size for token in tokens]
        return {"input_ids": torch.tensor([token_ids])}
