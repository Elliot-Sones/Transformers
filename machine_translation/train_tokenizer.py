"""
Train a BPE tokenizer for EN→FR machine translation.

This script:
  1) Loads the EN-FR sentence pairs from CSVs
  2) Trains a WordPiece tokenizer on both languages
  3) Saves to JSON for use with HuggingFace tokenizers library

Usage:
  python -m machine_translation.train_tokenizer --out machine_translation/archive/tokenizer.json

Prerequisite: Run setup_data.py first to download and create train.csv/test.csv
"""

import argparse
import os
from typing import Iterator, List

import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer
from tokenizers.processors import TemplateProcessing


def iter_sentences(csv_paths: List[str], cols: List[str] = ["en", "fr"]) -> Iterator[str]:
    """Yield sentences from all specified columns in all CSVs."""
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping.")
            continue
        df = pd.read_csv(csv_path)
        for col in cols:
            if col in df.columns:
                for text in df[col].dropna().astype(str):
                    yield text


def train_tokenizer(
    train_csv: str,
    test_csv: str,
    output_path: str,
    vocab_size: int = 32000,
    min_frequency: int = 2,
) -> Tokenizer:
    """Train a WordPiece tokenizer on EN-FR data."""
    
    # Initialize tokenizer with WordPiece model
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    # Define special tokens
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    
    # Create trainer
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )
    
    # Collect all sentences
    csv_paths = [train_csv, test_csv]
    sentences = list(iter_sentences(csv_paths, cols=["en", "fr"]))
    
    if len(sentences) == 0:
        raise RuntimeError(
            f"No sentences found in {csv_paths}. "
            "Run setup_data.py first to download training data."
        )
    
    print(f"Training tokenizer on {len(sentences):,} sentences...")
    
    # Train on iterator
    tokenizer.train_from_iterator(sentences, trainer=trainer)
    
    # Add post-processor for [CLS] and [SEP] handling (optional but clean)
    tokenizer.post_processor = TemplateProcessing(
        single="$A",
        pair="$A [SEP] $B:1",
        special_tokens=[
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save tokenizer
    tokenizer.save(output_path)
    print(f"Tokenizer saved to {output_path}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Quick validation
    test_en = "Hello, how are you today?"
    test_fr = "Bonjour, comment allez-vous aujourd'hui?"
    print(f"\nTest encoding:")
    print(f"  EN: '{test_en}' -> {tokenizer.encode(test_en).tokens}")
    print(f"  FR: '{test_fr}' -> {tokenizer.encode(test_fr).tokens}")
    
    return tokenizer


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train tokenizer for EN-FR translation")
    p.add_argument(
        "--train_csv",
        type=str,
        default=os.path.join("machine_translation", "archive", "train.csv"),
        help="Path to training CSV",
    )
    p.add_argument(
        "--test_csv",
        type=str,
        default=os.path.join("machine_translation", "archive", "test.csv"),
        help="Path to test CSV",
    )
    p.add_argument(
        "--out",
        type=str,
        default=os.path.join("machine_translation", "archive", "tokenizer.json"),
        help="Output path for tokenizer",
    )
    p.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Target vocabulary size",
    )
    p.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="Minimum token frequency to include in vocab",
    )
    return p


def main():
    args = build_argparser().parse_args()
    train_tokenizer(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        output_path=args.out,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )


if __name__ == "__main__":
    main()
