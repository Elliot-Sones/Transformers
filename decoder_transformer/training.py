import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import time
import os
import argparse
import signal
import copy
import urllib.request
from datetime import datetime
from contextlib import nullcontext


def ensure_data(data_path='archive/train.csv'):
    """Download Tiny Shakespeare if not present."""
    if not os.path.exists(data_path):
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        print(f'Downloading Tiny Shakespeare from {url}...')
        urllib.request.urlretrieve(url, data_path)
        print(f'Saved to {data_path}')
    return data_path


# ========== Hyperparameters
batch_size = 64
block_size = 256 # We will predict the 257 token on the basis of the 256 before that now!
max_iters = 5000
eval_interval = 1000
learning_rate = 3e-4 # Bring down the learning rate
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
eval_iters = 50
n_embd = 384 # 384 / 6 = 64
n_head = 6
n_layer = 6
dropout = 0.1
label_smoothing = 0.05
ema_decay = 0.999
use_ema_for_eval = True
use_sdpa = True
use_compile = True

# LR schedule (cosine with warmup)
warmup_iters = 200
lr_decay_iters = max_iters
min_lr = 1e-4
# ==========================
torch.manual_seed(1337)

# Dataset
data_path = ensure_data('archive/train.csv')
with open(data_path, 'r', encoding='UTF-8') as f:
    text = f.read()

# Tokenizer
chars = sorted(list(set(text)))
vocab_size = len(chars)
lookup_table_in = { ch:i for i,ch in enumerate(chars)}
lookup_table_out = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [lookup_table_in[c] for c in s] # Encoder
decode = lambda l: ''.join([lookup_table_out[i] for i in l]) # Decoder
data = torch.tensor(encode(text), dtype=torch.long)

# Train and Test Split
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Data Loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Loss
@torch.no_grad()
def estimate_loss():
    out = {}
    eval_model = (ema_model if (use_ema_for_eval and 'ema_model' in globals() and ema_model is not None) else model)
    eval_model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = eval_model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# =========== Transformer Components:

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Keep for reference but SDPA handles causal mask internally
        # self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, hs)
        q = self.query(x) # (B, T, hs)
        v = self.value(x) # (B, T, hs)
        if use_sdpa:
            # Use PyTorch SDPA; add a head dimension of size 1
            qh = q.unsqueeze(1)  # (B, 1, T, hs)
            kh = k.unsqueeze(1)  # (B, 1, T, hs)
            vh = v.unsqueeze(1)  # (B, 1, T, hs)
            out = F.scaled_dot_product_attention(
                qh, kh, vh,
                attn_mask=None,
                dropout_p=dropout if self.training else 0.0,
                is_causal=True,
            )  # (B, 1, T, hs)
            out = out.squeeze(1) # (B, T, hs)
        else:
            wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
            # Causal mask
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            wei = wei.masked_fill(~mask, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        # Added the possibility to add heads per parameter and loop. That's it.
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout) # <----- More Dropout!

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out)) # <----- More Dropout!
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(approximate='tanh'),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout), # <----- More Dropout!
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# We now don't have a BigramLanguage anymore
class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Added the possibility to add heads per parameter and loop. That's it.
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        # Weight tying: improves perplexity and reduces params slightly
        self.lm_head.weight = self.token_embedding_table.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # Compute CE loss (float32) with label smoothing for stability
            loss = F.cross_entropy(logits.float(), targets, label_smoothing=label_smoothing)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
# Train =============================
model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
print(device)

# Optionally compile the model for speed (requires PyTorch 2.x)
if use_compile:
    try:
        model = torch.compile(model)
        m = model  # keep reference consistent
        print('torch.compile: enabled')
    except Exception as e:
        print(f'warning: torch.compile failed: {e}')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1, betas=(0.9, 0.95))

# autocast context for mixed precision (CUDA or MPS)
if device == 'cuda':
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16)
elif device == 'mps':
    ctx = torch.amp.autocast(device_type='mps', dtype=torch.float16)
else:
    ctx = nullcontext()

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / max(1, warmup_iters)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

log_window = 100
t_last = time.time()

# ========== Checkpointing, resume, and interrupt handling
def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d

def _checkpoint_dir(out_dir):
    return _ensure_dir(out_dir if out_dir else os.path.join('assets', 'checkpoints'))

def save_ckpt(path, step):
    ckpt = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iter': step,
        'meta': {
            'chars': chars,
            'vocab_size': vocab_size,
            'n_embd': n_embd,
            'n_head': n_head,
            'n_layer': n_layer,
            'block_size': block_size,
            'dropout': dropout,
            'label_smoothing': label_smoothing,
            'ema_decay': ema_decay,
        }
    }
    # Include EMA weights if available
    if 'ema_model' in globals() and ema_model is not None:
        try:
            ckpt['ema_state_dict'] = ema_model.state_dict()
        except Exception:
            pass
    torch.save(ckpt, path)

def auto_latest_path(out_dir):
    return os.path.join(_checkpoint_dir(out_dir), 'latest.pt')

def timed_step_path(out_dir, step):
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    return os.path.join(_checkpoint_dir(out_dir), f'gpt-{ts}-step{step}.pt')

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint if available or from --ckpt')
parser.add_argument('--ckpt', type=str, default=None, help='Specific checkpoint path to resume from')
parser.add_argument('--save_interval', type=int, default=0, help='Steps between periodic checkpoints (0 to disable)')
parser.add_argument('--save_twice', dest='save_twice', action='store_true', default=True, help='Save exactly twice at 1/3 and 2/3 progress')
parser.add_argument('--no_save_twice', dest='save_twice', action='store_false', help='Disable the two-milestone saves')
parser.add_argument('--out_dir', type=str, default=os.path.join('assets', 'checkpoints'), help='Directory to write checkpoints')
try:
    args, _unknown = parser.parse_known_args()
except SystemExit:
    class _A: pass
    args = _A()
    args.resume = False
    args.ckpt = None
    args.save_interval = 0
    args.out_dir = os.path.join('assets', 'checkpoints')

start_iter = 0
if args.resume:
    resume_path = args.ckpt if args.ckpt else (auto_latest_path(args.out_dir) if os.path.exists(auto_latest_path(args.out_dir)) else None)
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        try:
            optimizer.load_state_dict(state['optimizer_state_dict'])
        except Exception as e:
            print(f"warning: could not load optimizer state: {e}")
        start_iter = int(state.get('iter', -1)) + 1
        if start_iter < 0:
            start_iter = 0
        print(f"Resumed at step {start_iter}")
    else:
        print("--resume requested but no checkpoint found; starting fresh.")

# Initialize EMA model after potential resume has loaded model
ema_model = None
if ema_decay and ema_decay > 0.0:
    ema_model = copy.deepcopy(model).to(device)
    for p in ema_model.parameters():
        p.requires_grad_(False)

# Milestone saves at ~1/3 and ~2/3 of max_iters
milestones = sorted({max(1, round(max_iters/3)), max(1, round(2*max_iters/3))})
print(f"Milestone checkpoints planned at steps: {milestones}")

interrupt_flag = {'hit': False}

def _handle_sigint(signum, frame):
    interrupt_flag['hit'] = True
    print("\nCtrl+C detected; will save checkpoint at next safe point...")

signal.signal(signal.SIGINT, _handle_sigint)

for iter in range(start_iter, max_iters):

    # every once in a while evaluate the loss on train and val sets (skip step 0)
    if iter > 0 and (iter % eval_interval == 0 or iter == max_iters - 1):
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # update learning rate via schedule
    lr = get_lr(iter)
    for g in optimizer.param_groups:
        g['lr'] = lr

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    with ctx:
        logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # EMA update
    if ema_model is not None:
        with torch.no_grad():
            msd = model.state_dict()
            for (k, v_ema) in ema_model.state_dict().items():
                v = msd[k]
                if v_ema.dtype.is_floating_point:
                    v_ema.mul_(ema_decay).add_(v, alpha=(1.0 - ema_decay))

    # progress logging: avg ms/iter over last window and ETA
    if (iter + 1) % log_window == 0:
        t_now = time.time()
        ms_per_iter = (t_now - t_last) * 1000.0 / log_window
        t_last = t_now
        remaining = max_iters - (iter + 1)
        eta_min = (remaining * ms_per_iter) / 1000.0 / 60.0
        print(f"~{ms_per_iter:.1f} ms/iter, ETA {eta_min:.1f} min, lr {lr:.2e}")

    # milestone and/or periodic checkpoint save
    do_milestone = args.save_twice and (iter in milestones)
    do_periodic = (args.save_interval and args.save_interval > 0 and (iter % args.save_interval == 0))
    if iter > 0 and (do_milestone or do_periodic):
        latest = auto_latest_path(args.out_dir)
        step_path = timed_step_path(args.out_dir, iter)
        try:
            save_ckpt(latest, iter)
            save_ckpt(step_path, iter)
            which = 'milestone' if do_milestone and not do_periodic else ('periodic' if do_periodic and not do_milestone else 'periodic+milestone')
            print(f"Saved {which} checkpoint at step {iter} -> {latest} and {step_path}")
        except Exception as e:
            print(f"warning: failed to save checkpoint at step {iter}: {e}")

    # handle Ctrl+C gracefully: save and exit
    if interrupt_flag['hit']:
        latest = auto_latest_path(args.out_dir)
        try:
            save_ckpt(latest, iter)
            print(f"Checkpoint saved on interrupt at step {iter} -> {latest}")
        except Exception as e:
            print(f"warning: failed to save interrupt checkpoint: {e}")
        break

# ========== Save final checkpoint and quick sample (if not interrupted)
if not interrupt_flag['hit']:
    # Evaluate final losses for reference
    losses = estimate_loss()
    print(f"final: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Save model checkpoint with meta and optimizer
    latest_path = auto_latest_path(args.out_dir)
    step_path = timed_step_path(args.out_dir, max_iters - 1)
    try:
        save_ckpt(latest_path, max_iters - 1)
        save_ckpt(step_path, max_iters - 1)
        print(f"Saved checkpoint to {latest_path}\nSnapshot at {step_path}")
    except Exception as e:
        print(f"warning: failed to save final checkpoint: {e}")

    # Emit a short sample to verify end-to-end
    model.eval()
    with torch.no_grad():
        # start from an empty context (first token index)
        start_idx = torch.zeros((1, 1), dtype=torch.long, device=device)
        out_idx = model.generate(start_idx, max_new_tokens=200)[0].tolist()
        sample_text = decode(out_idx)
        print("\n=== Sample (200 chars) ===")
        print(sample_text[:200])
        print("==========================\n")
