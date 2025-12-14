"""
Script to resize model embeddings for the new tokenizer.
Loads old checkpoint, expands embedding matrix, saves new checkpoint.
"""
import torch
import os

# Paths
OLD_CKPT = "checkpoints/mini/best.pt"
NEW_CKPT = "checkpoints/mini/best_resized.pt"
OLD_VOCAB_SIZE = 32000
NEW_VOCAB_SIZE = 32136  # 136 new tokens added

print(f"Loading checkpoint from {OLD_CKPT}...")
ckpt = torch.load(OLD_CKPT, map_location="cpu")
config = ckpt["config"]
state_dict = ckpt["model"]

print(f"Old vocab size in config: {config.get('src_vocab_size', 'N/A')}")
print(f"New vocab size: {NEW_VOCAB_SIZE}")

# Update config vocab sizes
config["src_vocab_size"] = NEW_VOCAB_SIZE
config["tgt_vocab_size"] = NEW_VOCAB_SIZE

# Find embedding layers to resize
embed_dim = config.get("embed_dim", 384)
print(f"Embedding dim: {embed_dim}")

# Resize encoder embedding
enc_emb_key = "encoder.emb.token_embedding.weight"
if enc_emb_key in state_dict:
    old_emb = state_dict[enc_emb_key]
    print(f"Encoder embedding shape: {old_emb.shape}")
    new_emb = torch.zeros(NEW_VOCAB_SIZE, embed_dim)
    new_emb[:OLD_VOCAB_SIZE] = old_emb
    # Initialize new tokens with small random values
    new_emb[OLD_VOCAB_SIZE:] = torch.randn(NEW_VOCAB_SIZE - OLD_VOCAB_SIZE, embed_dim) * 0.02
    state_dict[enc_emb_key] = new_emb
    print(f"Resized encoder embedding to: {new_emb.shape}")

# Resize decoder embedding
dec_emb_key = "decoder.emb.token_embedding.weight"
if dec_emb_key in state_dict:
    old_emb = state_dict[dec_emb_key]
    print(f"Decoder embedding shape: {old_emb.shape}")
    new_emb = torch.zeros(NEW_VOCAB_SIZE, embed_dim)
    new_emb[:OLD_VOCAB_SIZE] = old_emb
    new_emb[OLD_VOCAB_SIZE:] = torch.randn(NEW_VOCAB_SIZE - OLD_VOCAB_SIZE, embed_dim) * 0.02
    state_dict[dec_emb_key] = new_emb
    print(f"Resized decoder embedding to: {new_emb.shape}")

# Resize output projection (lm_head)
lm_head_key = "lm_head.weight"
if lm_head_key in state_dict:
    old_head = state_dict[lm_head_key]
    print(f"LM head shape: {old_head.shape}")
    new_head = torch.zeros(NEW_VOCAB_SIZE, embed_dim)
    new_head[:OLD_VOCAB_SIZE] = old_head
    new_head[OLD_VOCAB_SIZE:] = torch.randn(NEW_VOCAB_SIZE - OLD_VOCAB_SIZE, embed_dim) * 0.02
    state_dict[lm_head_key] = new_head
    print(f"Resized LM head to: {new_head.shape}")

# Check for bias in lm_head
lm_head_bias_key = "decoder.lm_head.bias"
if lm_head_bias_key in state_dict:
    old_bias = state_dict[lm_head_bias_key]
    print(f"LM head bias shape: {old_bias.shape}")
    new_bias = torch.zeros(NEW_VOCAB_SIZE)
    new_bias[:OLD_VOCAB_SIZE] = old_bias
    state_dict[lm_head_bias_key] = new_bias
    print(f"Resized LM head bias to: {new_bias.shape}")

# Save new checkpoint
ckpt["config"] = config
ckpt["model"] = state_dict
# Remove optimizer state since shapes changed
if "optimizer" in ckpt:
    del ckpt["optimizer"]

os.makedirs(os.path.dirname(NEW_CKPT), exist_ok=True)
torch.save(ckpt, NEW_CKPT)
print(f"\nSaved resized checkpoint to: {NEW_CKPT}")

# Verify it loads
print("\nVerifying new checkpoint...")
from mini_transformer import Seq2SeqConfig, Seq2Seq
new_cfg = Seq2SeqConfig(**config)
model = Seq2Seq(new_cfg)
model.load_state_dict(state_dict, strict=True)
print("✓ Checkpoint loads successfully!")
print(f"Model vocab size: {new_cfg.src_vocab_size}")
