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
from lcm_explo.domain.usecases.encode import encode_text_sonar
from lcm_explo.m2m_100 import M2M100EncoderModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ROC stories")
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

    args = parser.parse_args()

    file_system_embedding = FileSystemEmbeddingRepository(base_path=args.output_dir)

    roc_stories_dataset = load_dataset(
        "mintujupally/ROCStories",
        split="train",
    )

    sonar_encoder = M2M100EncoderModel.from_pretrained("cointegrated/SONAR_200_text_encoder_hf").to(args.device)
    sonar_tokenizer = NllbTokenizer.from_pretrained(
        "facebook/nllb-200-distilled-600M", src_lang="eng_Latn", tgt_lang="eng_Latn"
    )

    splitter = SaT("sat-3l")

    iter_dataset = iter(roc_stories_dataset)

    for index, article in tqdm(enumerate([next(iter_dataset) for _ in range(args.num_articles)])):
        title = f"roc_story_{index}"
        if title in file_system_embedding.list_documents():
            continue
        encoded_article = encode_text_sonar(
            article["text"],
            device=args.device,
            sonar_tokenizer=sonar_tokenizer,
            sonar_model=sonar_encoder,
            splitter=splitter,
            min_sentence_length=5,
        )
        document = DocumentEmbeddings(document_id=title, embeddings=encoded_article)

        file_system_embedding.save_document_embeddings(document)
