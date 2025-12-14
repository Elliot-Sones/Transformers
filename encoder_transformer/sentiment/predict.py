"""
Emotion Classification Inference Script

Predict emotions for text using a trained model.
Labels: sadness, joy, love, anger, fear, surprise

Usage:
    # Single text
    python encoder_transformer/sentiment/predict.py --text "I am so happy today!"

    # Interactive mode
    python encoder_transformer/sentiment/predict.py --interactive

    # Batch from file
    python encoder_transformer/sentiment/predict.py --input_file texts.txt --output_file predictions.csv
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from encoder_transformer.sentiment.train import EmotionClassifier, EMOTION_LABELS, detect_device

try:
    from tokenizers import Tokenizer
except ImportError:
    print("Please install: pip install tokenizers")
    sys.exit(1)


class EmotionPredictor:
    """Easy-to-use emotion prediction interface."""

    def __init__(self, checkpoint_path: str, device: torch.device = None):
        self.device = device or detect_device()
        print(f"Using device: {self.device}")

        # Load checkpoint
        print(f"Loading model from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        # Load tokenizer
        tokenizer_path = ckpt.get("tokenizer_path", "")
        if not tokenizer_path or not os.path.exists(tokenizer_path):
            # Try to find tokenizer in same directory as checkpoint
            ckpt_dir = Path(checkpoint_path).parent
            tokenizer_path = str(ckpt_dir / "tokenizer.json")

        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.pad_id = self.tokenizer.token_to_id("[PAD]") or 0
        self.max_len = ckpt["config"].get("max_seq_len", 128)
        self.labels = ckpt["config"].get("label_names", EMOTION_LABELS)

        # Build model from config
        config = ckpt["config"]
        self.model = EmotionClassifier(
            vocab_size=config["vocab_size"],
            num_classes=config["num_classes"],
            embed_dim=config["embed_dim"],
            ff_hidden_dim=config["ff_hidden_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            max_seq_len=config["max_seq_len"],
            dropout=config["dropout"],
            pool_method=config["pool_method"],
            pad_token_id=config["pad_token_id"],
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        print(f"Model loaded successfully!")
        print(f"  Best accuracy: {ckpt.get('best_acc', 'N/A')}")
        print(f"  Labels: {self.labels}")

    def _tokenize(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a single text."""
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids[:self.max_len]

        # Pad
        pad_len = self.max_len - len(ids)
        input_ids = ids + [self.pad_id] * pad_len
        attention_mask = [1] * len(ids) + [0] * pad_len

        return (
            torch.tensor([input_ids], dtype=torch.long, device=self.device),
            torch.tensor([attention_mask], dtype=torch.long, device=self.device),
        )

    @torch.no_grad()
    def predict(self, text: str) -> Tuple[str, float, dict]:
        """
        Predict emotion for a single text.

        Returns:
            (label, confidence, probabilities_dict)
        """
        input_ids, attention_mask = self._tokenize(text)
        _, logits = self.model(input_ids, attention_mask)

        probs = F.softmax(logits, dim=-1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()

        prob_dict = {self.labels[i]: probs[i].item() for i in range(len(self.labels))}

        return self.labels[pred_idx], confidence, prob_dict

    @torch.no_grad()
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float, dict]]:
        """Predict emotions for multiple texts."""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results


def main():
    parser = argparse.ArgumentParser(description="Predict emotions (sadness, joy, love, anger, fear, surprise)")
    parser.add_argument("--ckpt", type=str, default="encoder_transformer/sentiment/checkpoints/best.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--text", type=str, default="", help="Single text to classify")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--input_file", type=str, default="", help="File with texts (one per line)")
    parser.add_argument("--output_file", type=str, default="", help="Output CSV file")
    parser.add_argument("--device", type=str, default="", help="Device (cuda/cpu/mps)")

    args = parser.parse_args()

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = detect_device()

    # Load model
    predictor = EmotionPredictor(args.ckpt, device)

    # Single text mode
    if args.text:
        label, confidence, probs = predictor.predict(args.text)
        print(f"\nText: {args.text}")
        print(f"Emotion: {label.upper()} ({confidence:.1%})")
        print(f"\nAll probabilities:")
        for emotion in EMOTION_LABELS:
            bar = "█" * int(probs[emotion] * 20)
            print(f"  {emotion:10} {probs[emotion]:6.1%} {bar}")
        return

    # Batch mode
    if args.input_file:
        if not os.path.exists(args.input_file):
            print(f"File not found: {args.input_file}")
            return

        with open(args.input_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

        print(f"\nProcessing {len(texts)} texts...")
        results = predictor.predict_batch(texts)

        # Output results
        if args.output_file:
            import csv
            with open(args.output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["text", "emotion", "confidence"] + [f"prob_{e}" for e in EMOTION_LABELS])
                for text, (label, conf, probs) in zip(texts, results):
                    writer.writerow([text, label, f"{conf:.4f}"] + [f"{probs[e]:.4f}" for e in EMOTION_LABELS])
            print(f"Results saved to {args.output_file}")
        else:
            for text, (label, conf, probs) in zip(texts, results):
                print(f"{label.upper():10} ({conf:.1%}): {text[:70]}...")
        return

    # Interactive mode
    if args.interactive:
        print("\nInteractive Emotion Classification")
        print("Labels: sadness, joy, love, anger, fear, surprise")
        print("Type text and press Enter. Type 'quit' to exit.\n")

        while True:
            try:
                text = input("Enter text: ").strip()
                if text.lower() in ("quit", "exit", "q"):
                    break
                if not text:
                    continue

                label, confidence, probs = predictor.predict(text)
                print(f"  -> {label.upper()} ({confidence:.1%})")

                # Show all probabilities as a bar chart
                for emotion in EMOTION_LABELS:
                    bar = "█" * int(probs[emotion] * 20)
                    print(f"     {emotion:10} {probs[emotion]:5.1%} {bar}")
                print()

            except KeyboardInterrupt:
                print("\nExiting...")
                break

        return

    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
