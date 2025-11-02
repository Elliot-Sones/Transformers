import argparse
import os
import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------- Model definition (mirrors training.py) 
class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, hs)
        q = self.query(x) # (B, T, hs)
        v = self.value(x) # (B, T, hs)
        try:
            qh = q.unsqueeze(1)  # (B, 1, T, hs)
            kh = k.unsqueeze(1)  # (B, 1, T, hs)
            vh = v.unsqueeze(1)  # (B, 1, T, hs)
            out = F.scaled_dot_product_attention(
                qh, kh, vh,
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
            )  # (B, 1, T, hs)
            out = out.squeeze(1) # (B, T, hs)
        except Exception:
            # Fallback path without SDPA
            wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            wei = wei.masked_fill(~mask, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, head_size, dropout, block_size):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(approximate='tanh'),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, dropout, block_size)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, block_size=block_size, dropout=dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        # Weight tying
        self.lm_head.weight = self.token_embedding_table.weight

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def main():
    parser = argparse.ArgumentParser(description='Generate text from a trained miniGPT checkpoint')
    parser.add_argument('--ckpt', type=str, default=os.path.join('assets', 'checkpoints', 'latest.pt'), help='Path to checkpoint .pt')
    parser.add_argument('--prompt', type=str, default='', help='Prompt string (should use characters seen during training)')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Number of tokens to generate')
    parser.add_argument('--device', type=str, default=('mps' if torch.backends.mps.is_available() else 'cpu'), choices=['cpu', 'mps', 'cuda'], help='Device for inference')
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    device = args.device
    ckpt = torch.load(args.ckpt, map_location=device)
    meta = ckpt['meta']

    chars = meta['chars']
    vocab_size = meta['vocab_size']
    n_embd = meta['n_embd']
    n_head = meta['n_head']
    n_layer = meta['n_layer']
    block_size = meta['block_size']
    dropout = meta['dropout']

    lookup_table_in = {ch: i for i, ch in enumerate(chars)}
    lookup_table_out = {i: ch for i, ch in enumerate(chars)}

    def encode(s: str):
        # filter out any chars not in vocab to avoid KeyErrors
        return [lookup_table_in[c] for c in s if c in lookup_table_in]

    def decode(l):
        return ''.join([lookup_table_out[i] for i in l])

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        dropout=dropout,
    ).to(device)
    # Prefer EMA weights if available, else fall back to raw model weights
    state_key = 'ema_state_dict' if 'ema_state_dict' in ckpt and ckpt['ema_state_dict'] is not None else 'model_state_dict'

    def _normalize_compiled_keys(sd: dict):
        # Strip torch.compile wrapper prefix if present
        if any(k.startswith('_orig_mod.') for k in sd.keys()):
            return {k.replace('_orig_mod.', '', 1): v for k, v in sd.items()}
        return sd

    state_dict = _normalize_compiled_keys(ckpt[state_key])
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        # Surface a concise warning but continue; weights still load for inference
        print(f"Warning: load_state_dict mismatches. missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()

    if args.prompt:
        start_tokens = encode(args.prompt)
        if len(start_tokens) == 0:
            print("Warning: prompt contains no known characters; starting from empty context.")
    else:
        start_tokens = []

    if len(start_tokens) == 0:
        start = torch.zeros((1, 1), dtype=torch.long, device=device)
    else:
        start = torch.tensor([start_tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(start, max_new_tokens=args.max_new_tokens)
        text = decode(out[0].tolist())

    print("\n=== Generated ===")
    print(text)
    print("================")


if __name__ == '__main__':
    main()
