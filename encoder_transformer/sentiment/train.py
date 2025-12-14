"""
Emotion Classification Training Script

Trains a Transformer encoder for 6-way emotion classification.
Uses the dair-ai/emotion dataset: sadness, joy, love, anger, fear, surprise.

Ready for CUDA (4090 GPU) with mixed precision training.

Usage:
    python encoder_transformer/sentiment/train.py
    python encoder_transformer/sentiment/train.py --epochs 5 --batch_size 64
    python encoder_transformer/sentiment/train.py --resume --ckpt checkpoints/best.pt
"""

from __future__ import annotations

import os
import sys
import time
import math
import argparse
import random
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from encoder_transformer.encode import Encoder, EncoderConfig

# Try to import datasets (HuggingFace)
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: 'datasets' not installed. Run: pip install datasets")

# Try to import tokenizers
try:
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.trainers import WordPieceTrainer
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    print("Warning: 'tokenizers' not installed. Run: pip install tokenizers")


# Emotion labels from dair-ai/emotion dataset
EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def detect_device() -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Complexity indicators for curriculum learning
COMPLEXITY_CONJUNCTIONS = {"and", "but", "or", "because", "although", "however", "therefore", "moreover", "nevertheless", "furthermore", "whereas", "while", "since", "unless", "though", "despite", "yet"}


def compute_complexity(text: str) -> float:
    """
    Compute text complexity score (0=simple, 1=complex).
    
    Factors:
    - Word count (longer = more complex)
    - Vocabulary diversity (unique/total words)
    - Punctuation complexity (commas, semicolons, etc.)
    - Conjunction usage (indicates complex sentence structure)
    
    Returns:
        Float between 0 and 1
    """
    if not text or not text.strip():
        return 0.0
    
    words = text.lower().split()
    word_count = len(words)
    
    if word_count == 0:
        return 0.0
    
    # Factor 1: Length score (0-0.4)
    # Short (1-5 words) = 0, Long (50+ words) = 0.4
    length_score = min(word_count / 50, 1.0) * 0.4
    
    # Factor 2: Vocabulary diversity (0-0.25)
    unique_words = len(set(words))
    diversity_score = (unique_words / word_count) * 0.25
    
    # Factor 3: Punctuation complexity (0-0.2)
    punct_count = sum(1 for c in text if c in ',:;!?()"\'-')
    punct_score = min(punct_count / 10, 1.0) * 0.2
    
    # Factor 4: Conjunction usage (0-0.15)
    conj_count = sum(1 for w in words if w in COMPLEXITY_CONJUNCTIONS)
    conj_score = min(conj_count / 3, 1.0) * 0.15
    
    total = length_score + diversity_score + punct_score + conj_score
    return min(total, 1.0)


class EmotionClassifier(nn.Module):
    """
    Transformer Encoder + Classification Head for Emotion Classification.

    Architecture:
        Input tokens -> Encoder -> Pool (mean or [CLS]) -> Classifier -> logits
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 6,
        embed_dim: int = 256,
        ff_hidden_dim: int = 1024,
        num_heads: int = 4,
        num_layers: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        pool_method: str = "mean",  # "mean" or "cls"
        pad_token_id: int = 0,
    ):
        super().__init__()

        self.pool_method = pool_method
        self.pad_token_id = pad_token_id

        # Encoder configuration
        encoder_config = EncoderConfig(
            src_vocab_size=vocab_size,
            embed_dim=embed_dim,
            ff_hidden_dim=ff_hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_position_embeddings=max_seq_len,
            pad_token_id=pad_token_id,
            dropout=dropout,
        )

        self.encoder = Encoder(encoder_config)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        # Store config for checkpointing
        self.config = {
            "vocab_size": vocab_size,
            "num_classes": num_classes,
            "embed_dim": embed_dim,
            "ff_hidden_dim": ff_hidden_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "max_seq_len": max_seq_len,
            "dropout": dropout,
            "pool_method": pool_method,
            "pad_token_id": pad_token_id,
            "label_names": EMOTION_LABELS,
        }

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Args:
            input_ids: [B, S] token ids
            attention_mask: [B, S] 1=real token, 0=padding
            labels: [B] class labels (optional, for computing loss)

        Returns:
            (loss, logits) - loss is None if labels not provided
        """
        # Encode: [B, S, D]
        hidden = self.encoder(input_ids, attention_mask)

        # Pool to single vector per sequence: [B, D]
        if self.pool_method == "cls":
            # Use first token (assumes [CLS] at position 0)
            pooled = hidden[:, 0, :]
        else:
            # Mean pooling over non-padded tokens
            mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, S, 1]
            sum_hidden = (hidden * mask_expanded).sum(dim=1)  # [B, D]
            lengths = mask_expanded.sum(dim=1).clamp(min=1)  # [B, 1]
            pooled = sum_hidden / lengths

        # Classify: [B, num_classes]
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return loss, logits


class CSVEmotionDataset(Dataset):
    """Load emotion data from local CSV file (text, label columns)."""

    def __init__(
        self,
        csv_path: str,
        tokenizer: Tokenizer,
        max_len: int = 64,
    ):
        import csv as csv_module
        
        print(f"Loading from {csv_path}...")
        with open(csv_path, 'r') as f:
            reader = csv_module.DictReader(f)
            rows = list(reader)
        
        self.texts = [r['text'] for r in rows]
        self.labels = [int(r['label']) for r in rows]
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = tokenizer.token_to_id("[PAD]") or 0

        print(f"Loaded {len(self)} examples")
        print(f"Labels: {EMOTION_LABELS}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        text = self.texts[idx]
        label = int(self.labels[idx])

        encoding = self.tokenizer.encode(text)
        ids = encoding.ids[:self.max_len]

        pad_len = self.max_len - len(ids)
        ids = ids + [self.pad_id] * pad_len
        mask = [1] * (self.max_len - pad_len) + [0] * pad_len

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long),
            label,
        )


class EmotionDataset(Dataset):

    """dair-ai/emotion dataset wrapper for 6-way emotion classification."""

    def __init__(
        self,
        split: str,
        tokenizer: Tokenizer,
        max_len: int = 128,
        limit: Optional[int] = None,
        max_words: Optional[int] = None,  # Filter to texts with <= max_words
        use_unsplit: bool = False,  # Use 400k unsplit dataset
        add_synthetic: bool = False,  # Add synthetic short examples
    ):
        assert HF_AVAILABLE, "Please install: pip install datasets"

        # Load dataset
        if use_unsplit and split == "train":
            print(f"Loading emotion dataset (unsplit - 400k examples)...")
            ds = load_dataset("dair-ai/emotion", "unsplit", split="train")
        else:
            print(f"Loading emotion dataset ({split} split)...")
            ds = load_dataset("dair-ai/emotion", split=split)

        texts = list(ds["text"])
        labels = list(ds["label"])  # 0-5 corresponding to EMOTION_LABELS

        # Filter by word count if specified
        if max_words and max_words > 0:
            filtered_texts = []
            filtered_labels = []
            for t, l in zip(texts, labels):
                if len(t.split()) <= max_words:
                    filtered_texts.append(t)
                    filtered_labels.append(l)
            print(f"Filtered to texts with ≤{max_words} words: {len(filtered_texts)}/{len(texts)} ({len(filtered_texts)/len(texts)*100:.1f}%)")
            texts = filtered_texts
            labels = filtered_labels

        # Add synthetic short examples for training
        if add_synthetic and split == "train":
            synthetic = self._get_synthetic_examples()
            texts.extend([s[0] for s in synthetic])
            labels.extend([s[1] for s in synthetic])
            print(f"Added {len(synthetic)} synthetic short examples")

        if limit and limit > 0:
            texts = texts[:limit]
            labels = labels[:limit]

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = tokenizer.token_to_id("[PAD]") or 0

        print(f"Loaded {len(self)} examples")
        print(f"Labels: {EMOTION_LABELS}")

    def _get_synthetic_examples(self) -> List[Tuple[str, int]]:
        """Generate synthetic short examples for each emotion."""
        # Label indices: sadness=0, joy=1, love=2, anger=3, fear=4, surprise=5
        synthetic = [
            # SADNESS (0)
            ("i am sad", 0), ("im sad", 0), ("feeling sad", 0), ("so sad", 0),
            ("i feel sad", 0), ("this is sad", 0), ("im crying", 0), ("i cried", 0),
            ("heartbroken", 0), ("im heartbroken", 0), ("feeling down", 0), ("im down", 0),
            ("depressed", 0), ("feeling depressed", 0), ("miserable", 0), ("im miserable", 0),
            ("lonely", 0), ("im lonely", 0), ("feeling lonely", 0), ("so lonely", 0),
            ("unhappy", 0), ("im unhappy", 0), ("feeling blue", 0), ("grief", 0),
            ("sorrowful", 0), ("dejected", 0), ("hopeless", 0), ("im hopeless", 0),
            ("devastated", 0), ("im devastated", 0), ("gloomy", 0), ("melancholy", 0),
            
            # JOY (1)
            ("i am happy", 1), ("im happy", 1), ("feeling happy", 1), ("so happy", 1),
            ("i feel happy", 1), ("im so happy", 1), ("very happy", 1), ("really happy", 1),
            ("joyful", 1), ("im joyful", 1), ("feeling joyful", 1), ("full of joy", 1),
            ("excited", 1), ("im excited", 1), ("so excited", 1), ("feeling excited", 1),
            ("delighted", 1), ("im delighted", 1), ("thrilled", 1), ("im thrilled", 1),
            ("cheerful", 1), ("im cheerful", 1), ("ecstatic", 1), ("overjoyed", 1),
            ("elated", 1), ("blissful", 1), ("content", 1), ("im content", 1),
            ("pleased", 1), ("im pleased", 1), ("glad", 1), ("im glad", 1),
            
            # LOVE (2)
            ("i love you", 2), ("love you", 2), ("i love this", 2), ("love this", 2),
            ("i love it", 2), ("love it", 2), ("i love him", 2), ("i love her", 2),
            ("feeling loved", 2), ("im in love", 2), ("so in love", 2), ("deeply in love", 2),
            ("adore you", 2), ("i adore you", 2), ("adore this", 2), ("i adore it", 2),
            ("cherish you", 2), ("i cherish you", 2), ("loving this", 2), ("loving it", 2),
            ("affectionate", 2), ("feeling affection", 2), ("fond of you", 2), ("im fond of you", 2),
            ("passionate", 2), ("feeling passionate", 2), ("devoted", 2), ("im devoted", 2),
            ("smitten", 2), ("im smitten", 2), ("enamored", 2), ("besotted", 2),
            
            # ANGER (3)
            ("i am angry", 3), ("im angry", 3), ("feeling angry", 3), ("so angry", 3),
            ("i hate you", 3), ("hate you", 3), ("i hate this", 3), ("hate this", 3),
            ("i hate it", 3), ("hate it", 3), ("furious", 3), ("im furious", 3),
            ("mad", 3), ("im mad", 3), ("so mad", 3), ("really mad", 3),
            ("pissed off", 3), ("im pissed", 3), ("annoyed", 3), ("im annoyed", 3),
            ("irritated", 3), ("im irritated", 3), ("frustrated", 3), ("im frustrated", 3),
            ("outraged", 3), ("im outraged", 3), ("enraged", 3), ("livid", 3),
            ("resentful", 3), ("bitter", 3), ("hostile", 3), ("seething", 3),
            
            # FEAR (4)
            ("i am scared", 4), ("im scared", 4), ("feeling scared", 4), ("so scared", 4),
            ("i am afraid", 4), ("im afraid", 4), ("feeling afraid", 4), ("very afraid", 4),
            ("terrified", 4), ("im terrified", 4), ("frightened", 4), ("im frightened", 4),
            ("anxious", 4), ("im anxious", 4), ("feeling anxious", 4), ("so anxious", 4),
            ("worried", 4), ("im worried", 4), ("feeling worried", 4), ("very worried", 4),
            ("nervous", 4), ("im nervous", 4), ("feeling nervous", 4), ("so nervous", 4),
            ("panicked", 4), ("im panicking", 4), ("horrified", 4), ("dreading", 4),
            ("uneasy", 4), ("apprehensive", 4), ("alarmed", 4), ("spooked", 4),
            
            # SURPRISE (5)
            ("wow", 5), ("omg", 5), ("oh my god", 5), ("what", 5),
            ("surprised", 5), ("im surprised", 5), ("so surprised", 5), ("really surprised", 5),
            ("shocked", 5), ("im shocked", 5), ("so shocked", 5), ("totally shocked", 5),
            ("amazed", 5), ("im amazed", 5), ("astonished", 5), ("im astonished", 5),
            ("stunned", 5), ("im stunned", 5), ("speechless", 5), ("im speechless", 5),
            ("unbelievable", 5), ("incredible", 5), ("no way", 5), ("cant believe it", 5),
            ("mind blown", 5), ("didnt expect that", 5), ("unexpected", 5), ("startled", 5),
            ("flabbergasted", 5), ("dumbfounded", 5), ("bewildered", 5), ("taken aback", 5),
        ]
        
        # Duplicate synthetic examples to give them more weight
        return synthetic * 10  # 1920 synthetic examples

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        text = self.texts[idx]
        label = int(self.labels[idx])

        # Tokenize
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids[:self.max_len]

        # Pad
        pad_len = self.max_len - len(ids)
        ids = ids + [self.pad_id] * pad_len
        mask = [1] * (self.max_len - pad_len) + [0] * pad_len

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long),
            label,
        )


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]):
    """Collate batch of (input_ids, attention_mask, label) tuples."""
    input_ids = torch.stack([b[0] for b in batch])
    attention_mask = torch.stack([b[1] for b in batch])
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return input_ids, attention_mask, labels


class CurriculumSampler(Sampler):
    """
    Curriculum learning sampler that starts with simple examples and gradually
    introduces more complex ones.
    
    Uses a pacing function to determine what fraction of the (sorted) dataset
    to use at each epoch.
    """
    
    def __init__(
        self,
        complexity_scores: List[float],
        epoch: int = 0,
        total_epochs: int = 5,
        pacing: str = "linear",  # linear, quadratic, or step
        min_fraction: float = 0.3,  # Start with at least 30% of data
    ):
        self.n = len(complexity_scores)
        # Sort indices by complexity (simple first)
        self.sorted_indices = sorted(range(self.n), key=lambda i: complexity_scores[i])
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.pacing = pacing
        self.min_fraction = min_fraction
        
    def _get_fraction(self) -> float:
        """Calculate what fraction of data to use this epoch."""
        progress = (self.epoch + 1) / self.total_epochs
        
        if self.pacing == "linear":
            fraction = self.min_fraction + (1 - self.min_fraction) * progress
        elif self.pacing == "quadratic":
            # Slower start, faster ramp at the end
            fraction = self.min_fraction + (1 - self.min_fraction) * (progress ** 2)
        elif self.pacing == "step":
            # 3 discrete steps
            if progress < 0.33:
                fraction = self.min_fraction
            elif progress < 0.66:
                fraction = 0.6
            else:
                fraction = 1.0
        else:
            fraction = 1.0  # fallback: use all data
        
        return min(fraction, 1.0)
    
    def set_epoch(self, epoch: int):
        """Update epoch for pacing calculation."""
        self.epoch = epoch
    
    def __iter__(self):
        fraction = self._get_fraction()
        n_samples = max(1, int(self.n * fraction))
        
        # Take the simplest n_samples examples and shuffle them
        indices = self.sorted_indices[:n_samples].copy()
        random.shuffle(indices)
        return iter(indices)
    
    def __len__(self):
        fraction = self._get_fraction()
        return max(1, int(self.n * fraction))


def compute_dataset_complexity(texts: List[str]) -> List[float]:
    """Compute complexity scores for all texts in a dataset."""
    print("Computing text complexity scores for curriculum learning...")
    scores = [compute_complexity(t) for t in texts]
    
    # Print statistics
    avg_score = sum(scores) / len(scores)
    min_score = min(scores)
    max_score = max(scores)
    print(f"  Complexity: min={min_score:.2f}, avg={avg_score:.2f}, max={max_score:.2f}")
    
    # Show examples
    sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i])
    print(f"  Simplest: '{texts[sorted_idx[0]][:60]}...' (score={scores[sorted_idx[0]]:.2f})")
    print(f"  Complex:  '{texts[sorted_idx[-1]][:60]}...' (score={scores[sorted_idx[-1]]:.2f})")
    
    return scores


def build_tokenizer(texts: List[str], vocab_size: int = 30000, save_path: str = None) -> Tokenizer:
    """Build a WordPiece tokenizer from texts."""
    assert TOKENIZERS_AVAILABLE, "Please install: pip install tokenizers"

    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )

    # Train on texts
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Add special token processing
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", tokenizer.token_to_id("[CLS]")),
                       ("[SEP]", tokenizer.token_to_id("[SEP]"))],
    )

    if save_path:
        tokenizer.save(save_path)
        print(f"Tokenizer saved to {save_path}")

    return tokenizer


def save_checkpoint(
    path: str,
    model: EmotionClassifier,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_acc: float,
    tokenizer_path: str,
):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": model.config,
        "step": step,
        "best_acc": best_acc,
        "tokenizer_path": tokenizer_path,
    }, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(path: str, device: torch.device) -> dict:
    """Load training checkpoint."""
    return torch.load(path, map_location=device)


@torch.no_grad()
def evaluate(
    model: EmotionClassifier,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 0,
) -> Tuple[float, float]:
    """Evaluate model on dataset. Returns (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for i, (input_ids, attention_mask, labels) in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        loss, logits = model(input_ids, attention_mask, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    model.train()
    avg_loss = total_loss / max(1, total_samples)
    accuracy = total_correct / max(1, total_samples)
    return avg_loss, accuracy


def train(args):
    """Main training loop."""

    # Device
    device = detect_device()
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    # Paths
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = str(out_dir / "tokenizer.json")

    # Build or load tokenizer
    if args.tokenizer_path and os.path.exists(args.tokenizer_path):
        print(f"Loading tokenizer from {args.tokenizer_path}")
        tokenizer_path = args.tokenizer_path
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        print("Building tokenizer from emotion training data...")
        train_ds = load_dataset("dair-ai/emotion", split="train", trust_remote_code=True)
        tokenizer = build_tokenizer(
            train_ds["text"][:args.tokenizer_texts],
            vocab_size=args.vocab_size,
            save_path=tokenizer_path,
        )

    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("[PAD]") or 0
    print(f"Vocabulary size: {vocab_size}")

    # Datasets
    print("\nPreparing datasets...")
    train_dataset = EmotionDataset(
        split="train",
        tokenizer=tokenizer,
        max_len=args.max_seq_len,
        limit=args.train_limit,
        max_words=args.max_words,
        use_unsplit=args.use_unsplit,
        add_synthetic=args.add_synthetic,
    )
    val_dataset = EmotionDataset(
        split="validation",
        tokenizer=tokenizer,
        max_len=args.max_seq_len,
        limit=args.val_limit,
        max_words=args.max_words,  # Filter val set too for consistency
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    
    # Curriculum learning setup
    curriculum_sampler = None
    if args.curriculum:
        print("\nCurriculum learning enabled!")
        complexity_scores = compute_dataset_complexity(train_dataset.texts)
        curriculum_sampler = CurriculumSampler(
            complexity_scores=complexity_scores,
            epoch=0,
            total_epochs=args.epochs,
            pacing=args.pacing,
            min_fraction=args.curriculum_min_fraction,
        )
        # Recreate train_loader with sampler (no shuffle when using sampler)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=curriculum_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )
        print(f"  Pacing: {args.pacing}, min_fraction: {args.curriculum_min_fraction}")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model = EmotionClassifier(
        vocab_size=vocab_size,
        num_classes=6,  # sadness, joy, love, anger, fear, surprise
        embed_dim=args.embed_dim,
        ff_hidden_dim=args.ff_hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        pool_method=args.pool_method,
        pad_token_id=pad_id,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params / 1e6:.2f}M")
    print(f"Emotion classes: {EMOTION_LABELS}")
    
    # 4090 GPU optimizations
    if device.type == "cuda":
        # Enable TF32 for Ampere/Ada GPUs (4090) - significant speedup
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled for faster matrix operations")
        
        # torch.compile for PyTorch 2.0+ (major speedup on 4090)
        if args.compile and hasattr(torch, 'compile'):
            print("Compiling model with torch.compile (this may take a minute)...")
            model = torch.compile(model, mode="reduce-overhead")
            print("Model compiled successfully!")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )

    # Warmup + cosine decay scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    best_acc = 0.0

    if args.resume and args.ckpt and os.path.exists(args.ckpt):
        print(f"\nResuming from {args.ckpt}")
        ckpt = load_checkpoint(args.ckpt, device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        global_step = ckpt.get("step", 0)
        best_acc = ckpt.get("best_acc", 0.0)
        start_epoch = global_step // len(train_loader)
        print(f"Resumed at step {global_step}, best_acc={best_acc:.4f}")

    # Mixed precision for faster training on GPU
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else torch.amp.autocast(device_type="cpu", enabled=False)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Total steps: {total_steps}, Warmup: {warmup_steps}")

    model.train()
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        
        # Update curriculum sampler for this epoch
        if curriculum_sampler is not None:
            curriculum_sampler.set_epoch(epoch)
            n_samples = len(curriculum_sampler)
            frac = n_samples / len(train_dataset)
            print(f"\nEpoch {epoch+1}: Using {n_samples}/{len(train_dataset)} examples ({frac:.0%})")

        for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Forward pass with mixed precision
            with autocast_ctx:
                loss, logits = model(input_ids, attention_mask, labels)

            # Backward pass
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            scheduler.step()
            global_step += 1

            # Track metrics
            epoch_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            epoch_correct += (preds == labels).sum().item()
            epoch_samples += labels.size(0)

            # Log progress
            if global_step % args.log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                avg_loss = epoch_loss / epoch_samples
                acc = epoch_correct / epoch_samples
                elapsed = time.time() - start_time
                eta = elapsed / global_step * (total_steps - global_step)
                print(f"Step {global_step}/{total_steps} | Loss: {avg_loss:.4f} | "
                      f"Acc: {acc:.4f} | LR: {lr:.2e} | ETA: {eta/60:.1f}m")

        # End of epoch evaluation
        train_loss = epoch_loss / epoch_samples
        train_acc = epoch_correct / epoch_samples
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                str(out_dir / "best.pt"),
                model, optimizer, global_step, best_acc, tokenizer_path,
            )
            print(f"  New best! Val Acc: {best_acc:.4f}")

        # Save periodic checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                str(out_dir / f"epoch_{epoch+1}.pt"),
                model, optimizer, global_step, best_acc, tokenizer_path,
            )

    # Save final checkpoint
    save_checkpoint(
        str(out_dir / "final.pt"),
        model, optimizer, global_step, best_acc, tokenizer_path,
    )

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")
    print(f"Best validation accuracy: {best_acc:.4f}")


def build_argparser():
    p = argparse.ArgumentParser(description="Train emotion classifier (6-way: sadness, joy, love, anger, fear, surprise)")

    # Data
    p.add_argument("--train_limit", type=int, default=0, help="Limit train examples (0=all)")
    p.add_argument("--val_limit", type=int, default=0, help="Limit val examples (0=all)")
    p.add_argument("--max_seq_len", type=int, default=64, help="Max sequence length (reduced for short texts)")
    p.add_argument("--max_words", type=int, default=10, help="Filter to texts with <= max_words (0=no filter)")
    p.add_argument("--use_unsplit", action="store_true", default=True, help="Use 400k unsplit dataset")
    p.add_argument("--no_unsplit", action="store_false", dest="use_unsplit", help="Use 16k split dataset")
    p.add_argument("--add_synthetic", action="store_true", default=True, help="Add synthetic short examples")
    p.add_argument("--no_synthetic", action="store_false", dest="add_synthetic", help="Disable synthetic examples")
    p.add_argument("--tokenizer_path", type=str, default="", help="Path to existing tokenizer")
    p.add_argument("--tokenizer_texts", type=int, default=10000, help="Texts to train tokenizer on")
    p.add_argument("--vocab_size", type=int, default=30000, help="Tokenizer vocab size")

    # Curriculum learning
    p.add_argument("--curriculum", action="store_true", default=True, help="Enable curriculum learning (default: on)")
    p.add_argument("--no_curriculum", action="store_false", dest="curriculum", help="Disable curriculum learning")
    p.add_argument("--pacing", type=str, default="linear", choices=["linear", "quadratic", "step"], 
                   help="Curriculum pacing function")
    p.add_argument("--curriculum_min_fraction", type=float, default=0.3, 
                   help="Start with this fraction of (simplest) data")

    # Model
    p.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    p.add_argument("--ff_hidden_dim", type=int, default=1024, help="FFN hidden dimension")
    p.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    p.add_argument("--num_layers", type=int, default=4, help="Number of encoder layers")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    p.add_argument("--pool_method", type=str, default="mean", choices=["mean", "cls"])

    # Training
    p.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    p.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    p.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--compile", action="store_true", default=True, help="Use torch.compile (4090 optimization)")
    p.add_argument("--no_compile", action="store_false", dest="compile", help="Disable torch.compile")

    # Logging & checkpoints
    p.add_argument("--log_interval", type=int, default=50, help="Log every N steps")
    p.add_argument("--save_interval", type=int, default=1, help="Save every N epochs")
    p.add_argument("--out_dir", type=str, default="encoder_transformer/sentiment/checkpoints")
    p.add_argument("--resume", action="store_true", help="Resume training")
    p.add_argument("--ckpt", type=str, default="", help="Checkpoint to resume from")

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
