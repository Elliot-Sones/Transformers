# encode.py
from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncoderConfig:
    # Vocabulary size for source language (set from tokenizer)
    src_vocab_size: int
    # Model dimensions
    embed_dim: int = 512
    ff_hidden_dim: int = 2048
    num_heads: int = 8
    num_layers: int = 6
    # Regularization
    dropout: float = 0.1
    # Max sequence length for positional embeddings
    max_position_embeddings: int = 1024
    # Special tokens
    pad_token_id: int = 0
    # Initialization scale (optional, small init helps stability)
    init_range: float = 0.02


class TokenPositionalEmbedding(nn.Module):
    """
    Token embedding + learned positional embedding.
    Shapes:
      - input_ids: [B, S]
      - return: [B, S, D]
    """
    def __init__(self, vocab_size: int, embed_dim: int,
                 max_position_embeddings: int, pad_token_id: int, dropout: float):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.pos_embedding = nn.Embedding(max_position_embeddings, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        # [S] absolute positions 0..S-1
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        return self.dropout(x)  # [B, S, D]


class MultiHeadSelfAttention(nn.Module):
    """
    Standard MHA (Q=K=V) with padding mask support.
    Shapes:
      - x: [B, S, D]
      - key_padding_mask: [B, S] with True for tokens to keep OR 1/0; we convert to bool keep mask
      - return: [B, S, D]
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.FloatTensor, key_padding_mask: torch.Tensor) -> torch.FloatTensor:
        B, S, D = x.shape

        # Project to multihead Q, K, V: [B, S, H*Hd] -> [B, H, S, Hd]
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores: [B, H, S, Hd] @ [B, H, Hd, S] -> [B, H, S, S]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Build broadcastable mask over keys dimension: [B, 1, 1, S]
        # key_padding_mask is 1/True for valid tokens; 0/False for PADs.
        if key_padding_mask.dtype != torch.bool:
            keep_mask = key_padding_mask != 0
        else:
            keep_mask = key_padding_mask
        keep_mask = keep_mask.unsqueeze(1).unsqueeze(1)  # [B,1,1,S]

        # Mask PAD keys by setting scores to a large negative value (excluded after softmax)
        attn_scores = attn_scores.masked_fill(~keep_mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values: [B, H, S, S] @ [B, H, S, Hd] -> [B, H, S, Hd]
        attn_output = torch.matmul(attn_weights, v)

        # Merge heads: [B, H, S, Hd] -> [B, S, H*Hd=D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    """
    Position-wise MLP applied to each position independently.
    Shapes:
      - x: [B, S, D] -> [B, S, D]
    """
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class EncoderBlock(nn.Module):
    """
    One Pre-LN encoder block: LN -> MHA -> resid, then LN -> FFN -> resid.
    """
    def __init__(self, embed_dim: int, num_heads: int, ff_hidden_dim: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_hidden_dim, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.FloatTensor, key_padding_mask: torch.Tensor) -> torch.FloatTensor:
        # Self-attention sub-layer (Pre-LN)
        attn_out = self.self_attn(self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + self.dropout1(attn_out)

        # Feedforward sub-layer (Pre-LN)
        ff_out = self.ff(self.ln2(x))
        x = x + self.dropout2(ff_out)
        return x


class Encoder(nn.Module):
    """
    Full encoder: embeddings -> N blocks -> final LayerNorm.
    Forward signature:
      encoder_hidden_states = Encoder(config)(src_input_ids, src_attention_mask)
    """
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        assert config.embed_dim % config.num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embeddings = TokenPositionalEmbedding(
            vocab_size=config.src_vocab_size,
            embed_dim=config.embed_dim,
            max_position_embeddings=config.max_position_embeddings,
            pad_token_id=config.pad_token_id,
            dropout=config.dropout,
        )

        self.layers = nn.ModuleList([
            EncoderBlock(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                ff_hidden_dim=config.ff_hidden_dim,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])
        self.final_ln = nn.LayerNorm(config.embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_range)
            # Respect padding index: keep pad vectors near zero
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0.0)

    @torch.no_grad()
    def _ensure_mask_dtype(self, mask: torch.Tensor) -> torch.Tensor:
        # Accept bool or 0/1. Return bool where True means "keep".
        return mask.bool() if mask.dtype != torch.bool else mask

    def forward(
        self,
        src_input_ids: torch.LongTensor,       # [B, S]
        src_attention_mask: torch.Tensor,      # [B, S] (1/True=token, 0/False=PAD)
    ) -> torch.FloatTensor:
        x = self.embeddings(src_input_ids)     # [B, S, D]
        keep_mask = self._ensure_mask_dtype(src_attention_mask)

        for layer in self.layers:
            x = layer(x, key_padding_mask=keep_mask)

        x = self.final_ln(x)
        x = x * keep_mask.unsqueeze(-1)
        return x    # [B, S, D]