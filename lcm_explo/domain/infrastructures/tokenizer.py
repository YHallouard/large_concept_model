from abc import ABC, abstractmethod

import torch


class Tokenizer(ABC):
    @abstractmethod
    def __call__(
        self, text: str, return_tensors: str = "pt", padding: bool = False, truncation: bool = False
    ) -> dict[str, torch.Tensor]:
        pass
