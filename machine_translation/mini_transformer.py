from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Seq2SeqConfig:
    # Vocab sizes (set by tokenizer)
    src_vocab_size: int
    tgt_vocab_size: int
    # Model dimensions (tiny defaults)
    embed_dim: int = 256
    ff_hidden_dim: int = 1024
    num_heads: int = 4
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    # Regularization and positions
    dropout: float = 0.1
    max_position_embeddings: int = 512
    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    # Weight tying and init
    tie_embeddings: bool = True
    init_range: float = 0.02


class TokenPositionalEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, max_position_embeddings: int, pad_token_id: int, dropout: float):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.pos_embedding = nn.Embedding(max_position_embeddings, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        b, s = input_ids.shape
        device = input_ids.device
        pos = torch.arange(s, device=device).unsqueeze(0).expand(b, s)
        x = self.token_embedding(input_ids) + self.pos_embedding(pos)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Flexible MHA for self- or cross-attention.
    - If kv is None: uses x as K,V (self-attn).
    - Supports causal mask and key padding masks.
    Shapes:
      x: [B, T, D], kv: [B, S, D] or None
      key_padding_mask: [B, S] (True/1 = keep), causal: bool
      returns: [B, T, D]
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.FloatTensor,
        kv: Optional[torch.FloatTensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.FloatTensor:
        B, T, D = x.shape
        if kv is None:
            kv = x
        _, S, _ = kv.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,T,Hd]
        k = self.k_proj(kv).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,S,Hd]
        v = self.v_proj(kv).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,S,Hd]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,T,S]

        # Key padding mask over S dimension
        if key_padding_mask is not None:
            keep = key_padding_mask.bool().unsqueeze(1).unsqueeze(1)  # [B,1,1,S]
            attn_scores = attn_scores.masked_fill(~keep, float("-inf"))

        # Causal mask over T x S if self-attn and causal requested (only mask future positions)
        if causal:
            # Guard: causal is intended for self-attention (T==S)
            assert T == S, "Causal=True assumes self-attention (T==S)."
            # Build upper-triangular mask [T,S] where j > i are masked
            causal_mask = torch.ones((T, S), device=attn_scores.device, dtype=torch.bool).triu(1)
            attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        # If an entire row was masked, softmax(-inf) becomes NaN; zero those rows
        all_masked = torch.isinf(attn_scores).all(dim=-1, keepdim=True)  # [B,H,T,1]
        attn_weights = torch.where(all_masked, torch.zeros_like(attn_weights), attn_weights)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)  # [B,H,T,Hd]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_hidden_dim: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_hidden_dim, dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.FloatTensor, key_padding_mask: torch.Tensor) -> torch.FloatTensor:
        x = x + self.drop1(self.self_attn(self.ln1(x), kv=None, key_padding_mask=key_padding_mask, causal=False))
        x = x + self.drop2(self.ff(self.ln2(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, config: Seq2SeqConfig):
        super().__init__()
        self.config = config
        self.emb = TokenPositionalEmbedding(
            vocab_size=config.src_vocab_size,
            embed_dim=config.embed_dim,
            max_position_embeddings=config.max_position_embeddings,
            pad_token_id=config.pad_token_id,
            dropout=config.dropout,
        )
        self.layers = nn.ModuleList([
            EncoderBlock(config.embed_dim, config.num_heads, config.ff_hidden_dim, config.dropout)
            for _ in range(config.num_encoder_layers)
        ])
        self.ln = nn.LayerNorm(config.embed_dim)

    def forward(self, src_ids: torch.LongTensor, src_mask: torch.Tensor) -> torch.FloatTensor:
        x = self.emb(src_ids)
        keep = src_mask.bool()
        for layer in self.layers:
            x = layer(x, keep)
        x = self.ln(x)
        return x * keep.unsqueeze(-1)


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_hidden_dim: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.drop2 = nn.Dropout(dropout)

        self.ln3 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_hidden_dim, dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.FloatTensor,
        tgt_key_padding_mask: torch.Tensor,
        enc_out: torch.FloatTensor,
        src_key_padding_mask: torch.Tensor,
    ) -> torch.FloatTensor:
        # Masked self-attn
        x = x + self.drop1(self.self_attn(self.ln1(x), kv=None, key_padding_mask=tgt_key_padding_mask, causal=True))
        # Cross-attn
        x = x + self.drop2(self.cross_attn(self.ln2(x), kv=enc_out, key_padding_mask=src_key_padding_mask, causal=False))
        # FFN
        x = x + self.drop3(self.ff(self.ln3(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, config: Seq2SeqConfig):
        super().__init__()
        self.config = config
        self.emb = TokenPositionalEmbedding(
            vocab_size=config.tgt_vocab_size,
            embed_dim=config.embed_dim,
            max_position_embeddings=config.max_position_embeddings,
            pad_token_id=config.pad_token_id,
            dropout=config.dropout,
        )
        self.layers = nn.ModuleList([
            DecoderBlock(config.embed_dim, config.num_heads, config.ff_hidden_dim, config.dropout)
            for _ in range(config.num_decoder_layers)
        ])
        self.ln = nn.LayerNorm(config.embed_dim)

    def forward(
        self,
        tgt_ids: torch.LongTensor,
        tgt_mask: torch.Tensor,
        enc_out: torch.FloatTensor,
        src_mask: torch.Tensor,
    ) -> torch.FloatTensor:
        x = self.emb(tgt_ids)
        tgt_keep = tgt_mask.bool()
        src_keep = src_mask.bool()
        for layer in self.layers:
            x = layer(x, tgt_keep, enc_out, src_keep)
        x = self.ln(x)
        return x * tgt_keep.unsqueeze(-1)


class Seq2Seq(nn.Module):
    """
    Tiny Transformer encoder-decoder.
    Forward returns (loss, logits) if labels provided; else (None, logits).
    """
    def __init__(self, config: Seq2SeqConfig):
        super().__init__()
        assert config.embed_dim % config.num_heads == 0, "embed_dim must be divisible by num_heads"
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.lm_head = nn.Linear(config.embed_dim, config.tgt_vocab_size, bias=False)
        if config.tie_embeddings:
            # Tie with decoder token embeddings
            self.lm_head.weight = self.decoder.emb.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_range)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0.0)

    def _shift_right(self, ids: torch.LongTensor) -> torch.LongTensor:
        # Helper if needed (not used; we construct inputs/labels directly)
        b, t = ids.shape
        bos = torch.full((b, 1), self.config.bos_token_id, dtype=ids.dtype, device=ids.device)
        out = torch.cat([bos, ids[:, :-1]], dim=1)
        return out

    def forward(
        self,
        src_input_ids: torch.LongTensor,          # [B, S]
        src_attention_mask: torch.Tensor,         # [B, S] 1/True=keep, 0/False=pad
        tgt_input_ids: torch.LongTensor,          # [B, T] teacher-forced tokens (include BOS)
        tgt_attention_mask: torch.Tensor,         # [B, T]
        labels: Optional[torch.LongTensor] = None # [B, T] next-token labels aligned with tgt_input_ids
    ) -> Tuple[Optional[torch.Tensor], torch.FloatTensor]:
        # Early guard to prevent position embedding overflow
        assert src_input_ids.size(1) <= self.config.max_position_embeddings, "src length > max_position_embeddings"
        assert tgt_input_ids.size(1) <= self.config.max_position_embeddings, "tgt length > max_position_embeddings"
        # Sanity checks on mask dtypes and target length
        assert src_attention_mask.dtype in (torch.bool, torch.long), "src_attention_mask must be bool or long"
        assert tgt_attention_mask.dtype in (torch.bool, torch.long), "tgt_attention_mask must be bool or long"
        T = tgt_input_ids.size(1)
        assert T > 0, "tgt_input_ids must have length > 0 (should include BOS)"
        enc_out = self.encoder(src_input_ids, src_attention_mask)  # [B,S,D]
        dec_out = self.decoder(tgt_input_ids, tgt_attention_mask, enc_out, src_attention_mask)  # [B,T,D]
        logits = self.lm_head(dec_out)  # [B,T,V]

        loss = None
        if labels is not None:
            # Compute loss ignoring PAD labels
            vocab = logits.size(-1)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab),
                labels.reshape(-1),
                ignore_index=self.config.pad_token_id,
            )
        return loss, logits

    @torch.no_grad()
    def greedy_generate(
        self,
        src_input_ids: torch.LongTensor,
        src_attention_mask: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Greedy or sampled decoding with per-example EOS stopping.
        Returns token ids of shape [B, T_out]. Caller can strip BOS/EOS.
        """
        device = src_input_ids.device
        B = src_input_ids.size(0)
        enc_out = self.encoder(src_input_ids, src_attention_mask)

        ys = torch.full((B, 1), self.config.bos_token_id, dtype=torch.long, device=device)
        ys_mask = torch.ones((B, 1), dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            dec_out = self.decoder(ys, ys_mask, enc_out, src_attention_mask)
            logits = self.lm_head(dec_out[:, -1:, :])  # [B,1,V]

            if temperature and temperature > 0:
                logits = logits / max(temperature, 1e-6)
                if top_k is not None and top_k > 0:
                    k = min(int(top_k), logits.size(-1))
                    topk = torch.topk(logits, k=k, dim=-1)
                    thresh = topk.values[..., -1:].expand_as(logits)
                    logits = torch.where(logits >= thresh, logits, torch.full_like(logits, float("-inf")))
                probs = logits.softmax(dim=-1)
                next_ids = torch.multinomial(probs.squeeze(1), num_samples=1)  # [B,1]
            else:
                next_ids = torch.argmax(logits, dim=-1)  # [B,1]

            # Lock EOS for rows already finished
            eos = torch.full_like(next_ids, self.config.eos_token_id)
            next_ids = torch.where(finished.unsqueeze(1), eos, next_ids)

            ys = torch.cat([ys, next_ids], dim=1)
            ys_mask = torch.cat([ys_mask, torch.ones_like(next_ids)], dim=1)
            finished |= (next_ids.squeeze(1) == self.config.eos_token_id)
            if finished.all():
                break
        return ys

    @torch.no_grad()
    def decode_tokens(self, ys: torch.LongTensor) -> torch.LongTensor:
        """Strip BOS and pad everything after first EOS per example."""
        if ys.size(1) > 0 and (ys[:, 0] == self.config.bos_token_id).all():
            ys = ys[:, 1:]
        eos_mask = (ys == self.config.eos_token_id)
        if eos_mask.any():
            first_eos = eos_mask.float().argmax(dim=1)
            for b in range(ys.size(0)):
                if eos_mask[b].any():
                    cut = int(first_eos[b].item()) + 1
                    if cut < ys.size(1):
                        ys[b, cut:] = self.config.pad_token_id
        return ys


def prepare_decoder_inputs_and_labels(
    tgt_ids: torch.LongTensor,
    pad_token_id: int,
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """
    Given full target with BOS...EOS...PAD, produce:
      - decoder_input_ids (shifted right, starts with BOS)
      - decoder_attention_mask (1 for non-PAD)
      - labels (next tokens; PAD where ignored)
    Shapes: all [B, T-1] relative to original length if you trim EOS; here keep same length.
    """
    b, t = tgt_ids.shape
    attn = (tgt_ids != pad_token_id).long()
    # Inputs: everything except last token
    dec_in = tgt_ids[:, :-1].contiguous()
    dec_mask = attn[:, :-1].contiguous()
    # Labels: everything except first token
    labels = tgt_ids[:, 1:].contiguous()
    # Ensure PAD label where masked
    labels = labels * dec_mask + (1 - dec_mask) * pad_token_id
    return dec_in, dec_mask, labels
