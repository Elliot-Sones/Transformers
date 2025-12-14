"""
Train the encoder MLM model for word prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os

# Import our encoder
try:
    from ..encode import Encoder, EncoderConfig
except Exception:
    import pathlib, sys as _sys
    _sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from encoder_transformer.encode import Encoder, EncoderConfig

class ProductionTokenizer:
    """Tokenizer for the production sentence."""
    
    def __init__(self):
        # Build vocabulary from the target sentence
        target_sentence = "This model create relationships between the words to learn what word is missing!"
        words = target_sentence.lower().split()
        
        # Create vocabulary
        self.vocab = {}
        self.vocab["[PAD]"] = 0
        self.vocab["[UNK]"] = 1  
        self.vocab["[MASK]"] = 2
        
        # Add all words from the target sentence
        for i, word in enumerate(set(words)):  # Remove duplicates
            self.vocab[word] = i + 3
        
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Vocabulary: {list(self.vocab.keys())}")
        
    def encode(self, text: str) -> list:
        words = text.lower().split()
        return [self.vocab.get(word, self.vocab["[UNK]"]) for word in words]
    
    def decode(self, ids: list) -> str:
        return " ".join([self.id_to_token.get(idx, "[UNK]") for idx in ids])

class MLMDataset(Dataset):
    """Dataset with the target sentence repeated for training."""
    
    def __init__(self, max_len: int = 16):
        self.max_len = max_len
        
        # Target sentence
        self.target_sentence = "This model create relationships between the words to learn what word is missing!"
        
        # Repeat sentence many times for training
        self.texts = []
        for _ in range(1000):  # 1000 training samples
            self.texts.append(self.target_sentence)
        
        # Create tokenizer
        self.tokenizer = ProductionTokenizer()
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text)
        
        # Pad/truncate to max_len
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens = tokens + [self.tokenizer.vocab["[PAD]"]] * (self.max_len - len(tokens))
        
        return torch.tensor(tokens, dtype=torch.long)

def create_mlm_targets(input_ids: torch.Tensor, mask_id: int, pad_id: int, mask_prob: float):
    """Create masked language modeling targets for training."""
    device = input_ids.device
    B, S = input_ids.shape
    labels = input_ids.clone()
    
    # Only mask non-padding tokens
    is_token = input_ids != pad_id
    mask = (torch.rand((B, S), device=device) < mask_prob) & is_token
    
    labels[~mask] = -100  # ignore non-masked
    
    # Replace masked tokens with [MASK]
    masked_input = input_ids.clone()
    masked_input[mask] = mask_id
    
    return masked_input, labels

class ProductionMLM(nn.Module):
    """Production MLM for word prediction."""
    
    def __init__(self, vocab_size: int):
        super().__init__()
        
        # Production-ready encoder
        encoder_config = EncoderConfig(
            src_vocab_size=vocab_size,
            embed_dim=128,           # Small but effective
            ff_hidden_dim=256,       # Small but effective
            num_heads=4,             # Good for relationships
            num_layers=3,            # Enough depth
            max_position_embeddings=32,
            pad_token_id=0,
        )
        
        self.encoder = Encoder(encoder_config)
        
        # Output head for word prediction
        self.lm_head = nn.Linear(128, vocab_size)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        hidden = self.encoder(input_ids, attention_mask)
        logits = self.lm_head(hidden)
        return logits

def train_mlm():
    """Train the MLM model."""
    
    # Create dataset
    dataset = MLMDataset(max_len=16)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Create model
    model = ProductionMLM(vocab_size=dataset.tokenizer.vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    model.train()
    for epoch in range(20):
        total_loss = 0
        total_correct = 0
        total_masked = 0
        
        for batch in dataloader:
            input_ids = batch
            attention_mask = (input_ids != 0).float()
            
            # Create masked targets
            masked_input, labels = create_mlm_targets(
                input_ids, 
                mask_id=dataset.tokenizer.vocab["[MASK]"],
                pad_id=dataset.tokenizer.vocab["[PAD]"],
                mask_prob=0.2  # Mask 20% of tokens
            )
            
            # Forward pass
            logits = model(masked_input, attention_mask)
            loss = F.cross_entropy(logits.view(-1, dataset.tokenizer.vocab_size), labels.view(-1), ignore_index=-100)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            with torch.no_grad():
                preds = logits.argmax(-1)
                mask = labels != -100
                total_masked += mask.sum().item()
                if total_masked > 0:
                    total_correct += (preds[mask] == labels[mask]).sum().item()
            
            total_loss += loss.item()
        
        # Print progress
        avg_loss = total_loss / len(dataloader)
        accuracy = (total_correct / total_masked) if total_masked > 0 else 0
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f} ({accuracy*100:.1f}%)")
        
        if accuracy > 0.9:  # 90% accuracy target
            print(f"🎉 Target reached! Training accuracy: {accuracy*100:.1f}%")
            break
    
    # Save the trained model
    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_vocab': dataset.tokenizer.vocab,
        'target_sentence': "This model create relationships between the words to learn what word is missing!"
    }, 'models/production_model.pt')
    
    print("Model saved to models/production_model.pt")
    
    return model, dataset.tokenizer

if __name__ == "__main__":
    print("Training encoder MLM model...")
    model, tokenizer = train_mlm()
