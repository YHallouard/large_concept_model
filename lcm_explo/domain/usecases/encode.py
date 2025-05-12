import spacy
import torch

from lcm_explo.constant import SONAR_DIMENSIONS
from lcm_explo.domain.infrastructures.splitter import TextSplitter
from lcm_explo.domain.infrastructures.tokenizer import Tokenizer

nlp = spacy.load("en_core_web_sm")


def split_long_text(splitter: TextSplitter, text: str, min_sentence_length: int = 50) -> list[str]:
    splits = splitter.split(text)
    splits = [
        split.removesuffix(" ") for split in splits if split.replace(" ", "") != "" and len(split) > min_sentence_length
    ]
    return splits


def encode_text_sonar(
    text: str,
    device: str,
    sonar_tokenizer: Tokenizer,
    sonar_model: torch.nn.Module,
    splitter: TextSplitter,
    min_sentence_length: int = 50,
) -> torch.Tensor:
    embeddings = []
    with torch.no_grad():
        chunks = split_long_text(splitter, text, min_sentence_length=min_sentence_length)
        if not chunks:
            return torch.zeros((0, SONAR_DIMENSIONS), device=device)
        for chunk in chunks:
            inputs = sonar_tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = sonar_model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1))
    return torch.vstack(embeddings)


def encode_text_sonar_with_spacy(
    text: str,
    device: str,
    sonar_tokenizer: Tokenizer,
    sonar_model: torch.nn.Module,
    max_sentence_length: int = 75,
) -> torch.Tensor:
    embeddings = []
    with torch.no_grad():
        chunks = split_text(text, max_sentence_length)
        if not chunks:
            return torch.zeros((0, SONAR_DIMENSIONS), device=device)
        for chunk in chunks:
            inputs = sonar_tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = sonar_model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1))
    return torch.vstack(embeddings)


def spacy_segment(text: str) -> list[str]:
    doc = nlp(text)
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
