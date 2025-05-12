#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from datasets import load_dataset  # type: ignore[import-untyped]
from tqdm import tqdm
from transformers import NllbTokenizer  # type: ignore[import-untyped]
from wtpsplit import SaT  # type: ignore[import-untyped]

from lcm_explo.adapters.embedding_repository.file_system import FileSystemEmbeddingRepository
from lcm_explo.domain.models.documents import DocumentEmbeddings
from lcm_explo.domain.usecases.encode import encode_text_sonar, encode_text_sonar_with_spacy
from lcm_explo.m2m_100 import M2M100EncoderModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Wikipedia articles")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("embeddings"), help="Directory to save processed embeddings"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation",
    )
    parser.add_argument("--num-articles", type=int, default=5000, help="Number of articles to process")
    parser.add_argument("--splitter", type=str, default="sat", help="Splitter to use")

    args = parser.parse_args()

    file_system_embedding = FileSystemEmbeddingRepository(base_path=args.output_dir)

    wikipedia_dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
    )

    sonar_encoder = M2M100EncoderModel.from_pretrained("cointegrated/SONAR_200_text_encoder_hf").to(args.device)
    sonar_tokenizer = NllbTokenizer.from_pretrained(
        "facebook/nllb-200-distilled-600M", src_lang="eng_Latn", tgt_lang="eng_Latn"
    )

    splitter = SaT("sat-3l")

    iter_dataset = iter(wikipedia_dataset)

    already_computed_embeddings = file_system_embedding.list_documents()

    for article in tqdm([next(iter_dataset) for _ in range(args.num_articles)]):
        title = article["title"].replace("/", "_")
        if title in already_computed_embeddings:
            continue
        if args.splitter == "sat":
            encoded_article = encode_text_sonar(
                article["text"],
                device=args.device,
                sonar_tokenizer=sonar_tokenizer,
                sonar_model=sonar_encoder,
                splitter=splitter,
            )
        elif args.splitter == "spacy":
            encoded_article = encode_text_sonar_with_spacy(
                article["text"],
                device=args.device,
                sonar_tokenizer=sonar_tokenizer,
                sonar_model=sonar_encoder,
            )
        else:
            raise ValueError(f"Invalid splitter: {args.splitter}")  # noqa: TRY003

        document = DocumentEmbeddings(document_id=title, embeddings=encoded_article)

        file_system_embedding.save_document_embeddings(document)
