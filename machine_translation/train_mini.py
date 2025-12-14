import os
import time
import math
import argparse
import random
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tokenizers import Tokenizer

try:
    from machine_translation.mini_transformer import (
        Seq2SeqConfig,
        Seq2Seq,
        prepare_decoder_inputs_and_labels,
    )
except ModuleNotFoundError:
    from mini_transformer import (
        Seq2SeqConfig,
        Seq2Seq,
        prepare_decoder_inputs_and_labels,
    )


def detect_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CsvPairs(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer: Tokenizer,
        src_col: str = "en",
        tgt_col: str = "fr",
        max_src_len: int = 128,
        max_tgt_len: int = 128,
        limit: int = 0,
        shuffle: bool = True,
        bos_id: int = 2,
        eos_id: int = 3,
        chunk_size: int = 25000,
        verbose: bool = True,
    ):
        import pandas as pd

        assert os.path.exists(path), f"Missing CSV: {path}"

        # Setup special tokens and limits
        self.pad_id = tokenizer.token_to_id("[PAD]") or 0
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        # Prepare container and counters
        pairs: List[Tuple[List[int], List[int]]] = []
        processed = 0
        target_total = limit if (limit and limit > 0) else None

        # Use chunked CSV reading to avoid long, silent startup
        # Only keep needed columns to reduce memory
        chunks = pd.read_csv(
            path,
            usecols=[src_col, tgt_col],
            chunksize=min(chunk_size, limit) if (limit and limit > 0) else chunk_size,
        )

        for df in chunks:
            if target_total is not None and processed >= target_total:
                break

            if target_total is not None and processed + len(df) > target_total:
                df = df.iloc[: (target_total - processed)]

            # Coerce to strings
            src_texts = df[src_col].astype(str).tolist()
            tgt_texts = df[tgt_col].astype(str).tolist()

            # Tokenize this chunk
            src_enc = tokenizer.encode_batch(src_texts)
            tgt_enc = tokenizer.encode_batch(tgt_texts)

            # Build pairs
            for s, t in zip(src_enc, tgt_enc):
                sids = s.ids[:max_src_len]
                # Build target with BOS/EOS, then clip and ensure EOS
                tids = [self.bos_id] + t.ids + [self.eos_id]
                if len(tids) > max_tgt_len:
                    tids = tids[:max_tgt_len]
                    # Ensure last token is EOS
                    if tids[-1] != self.eos_id:
                        tids[-1] = self.eos_id
                if len(sids) == 0 or len(tids) < 2:
                    continue
                pairs.append((sids, tids))

            processed += len(df)
            if verbose and (processed // (chunk_size if chunk_size > 0 else 1)) >= 0:
                if target_total is not None:
                    print(f"Tokenized {processed}/{target_total} examples...")
                else:
                    print(f"Tokenized {processed} examples...")

        if shuffle:
            random.shuffle(pairs)
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        return self.pairs[idx]


def collate_batch(batch: List[Tuple[List[int], List[int]]], pad_id: int):
    # Compute max lens
    max_s = max(len(s) for s, _ in batch)
    max_t = max(len(t) for _, t in batch)
    B = len(batch)

    src = torch.full((B, max_s), pad_id, dtype=torch.long)
    src_mask = torch.zeros((B, max_s), dtype=torch.long)
    tgt_full = torch.full((B, max_t), pad_id, dtype=torch.long)

    for i, (s, t) in enumerate(batch):
        src[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        src_mask[i, : len(s)] = 1
        tgt_full[i, : len(t)] = torch.tensor(t, dtype=torch.long)

    # Build decoder inputs and labels
    dec_in, dec_mask, labels = prepare_decoder_inputs_and_labels(tgt_full, pad_id)
    return src, src_mask, dec_in, dec_mask, labels


def save_ckpt(path: str, model: Seq2Seq, step: int, config: Seq2SeqConfig, optimizer: Optional[torch.optim.Optimizer] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "step": step,
        "config": vars(config),
        "model": model.state_dict(),
    }
    if optimizer is not None:
        try:
            payload["optimizer"] = optimizer.state_dict()
        except Exception:
            # Optimizer state dict is best-effort; continue without it on errors
            pass
    torch.save(payload, path)


def evaluate(model: Seq2Seq, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for src, src_m, dec_in, dec_m, labels in loader:
            src = src.to(device)
            src_m = src_m.to(device)
            dec_in = dec_in.to(device)
            dec_m = dec_m.to(device)
            labels = labels.to(device)
            loss, _ = model(src, src_m, dec_in, dec_m, labels)
            losses.append(loss.item())
            if len(losses) >= 50:  # cap eval cost
                break
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def preview_samples(
    model: Seq2Seq,
    tok: Tokenizer,
    val_ds: CsvPairs,
    device: torch.device,
    n: int = 3,
    max_new_tokens: int = 64,
) -> List[Tuple[str, str, str]]:
    """Return up to n triples of (src_text, ref_text, hyp_text) for quick inspection."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(min(n, len(val_ds))):
            sids, tids = val_ds[i]
            src = torch.tensor([sids], dtype=torch.long, device=device)
            src_m = torch.ones_like(src)
            ys = model.greedy_generate(src, src_m, max_new_tokens=max_new_tokens)
            ys = model.decode_tokens(ys)[0].tolist()
            # Trim PADs
            ys = [t for t in ys if t != val_ds.pad_id]

            # Decode to strings
            src_text = tok.decode(sids)
            ref_text_ids = [t for t in tids if t not in (val_ds.bos_id, val_ds.eos_id, val_ds.pad_id)]
            ref_text = tok.decode(ref_text_ids)
            hyp_text = tok.decode(ys)
            out.append((src_text, ref_text, hyp_text))
    model.train()
    return out


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, default=os.path.join("machine_translation", "archive", "train.csv"))
    p.add_argument("--val_csv", type=str, default=os.path.join("machine_translation", "archive", "test.csv"))
    p.add_argument("--tokenizer_path", type=str, default=os.path.join("machine_translation", "archive", "tokenizer.json"))

    # Sequence lengths optimized for production (short translations)
    p.add_argument("--max_src_len", type=int, default=64)
    p.add_argument("--max_tgt_len", type=int, default=64)
    # Batch size optimized for 4090 (24GB VRAM)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--limit", type=int, default=250000, help="Limit of train examples (4090 can handle more)")
    p.add_argument("--val_limit", type=int, default=5000, help="Limit of validation examples")
    p.add_argument("--seed", type=int, default=1337)

    # Learning rate tuned for larger batch
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=4000)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--eval_interval", type=int, default=1000)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--save_interval", type=int, default=2000)
    p.add_argument("--print_samples", type=int, default=3, help="Print N sample translations at eval")
    p.add_argument("--patience_evals", type=int, default=10, help="Early stop if no val improvement for N evals")
    p.add_argument("--out_dir", type=str, default=os.path.join("machine_translation", "checkpoints", "mini"))
    # Resume options
    p.add_argument("--resume", action="store_true", help="Resume from out_dir/latest.pt if it exists")
    p.add_argument("--resume_from", type=str, default="", help="Path to checkpoint to resume from (overrides --resume)")
    p.add_argument("--reset_optim", action="store_true", help="When resuming, do not load optimizer state")
    p.add_argument("--reset_steps", action="store_true", help="When resuming, reset step counter to 0 (fresh LR schedule)")
    # Model size knobs (optimized for 4090 - deeper and wider)
    p.add_argument("--embed_dim", type=int, default=384)
    p.add_argument("--ff_hidden_dim", type=int, default=1536)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--enc_layers", type=int, default=6)
    p.add_argument("--dec_layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    # Mixed precision and compilation
    p.add_argument("--use_amp", action="store_true", default=True, help="Use automatic mixed precision (bf16)")
    p.add_argument("--compile", action="store_true", default=False, help="Use torch.compile for speedup")
    return p


def main():
    args = build_argparser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = detect_device()
    print(f"Using device: {device}")
    torch.set_float32_matmul_precision("high")

    # Tokenizer and special token ids
    tok = Tokenizer.from_file(args.tokenizer_path)
    vocab_size = tok.get_vocab_size()
    pad_id = tok.token_to_id("[PAD]") or 0
    bos_id = tok.token_to_id("[CLS]") or 2
    eos_id = tok.token_to_id("[SEP]") or 3

    # Datasets
    print("Building training dataset (tokenizing in chunks)...")
    train_ds = CsvPairs(
        args.train_csv, tok,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        limit=args.limit,
        bos_id=bos_id,
        eos_id=eos_id,
    )
    print("Building validation dataset (tokenizing in chunks)...")
    val_ds = CsvPairs(
        args.val_csv, tok,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        limit=args.val_limit,
        bos_id=bos_id,
        eos_id=eos_id,
        shuffle=False,
    )
    print(f"Datasets ready. Train: {len(train_ds)} examples | Val: {len(val_ds)} examples")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_batch(b, pad_id),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_batch(b, pad_id),
        drop_last=False,
    )

    # Model (may be overridden by resume)
    config = Seq2SeqConfig(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        ff_hidden_dim=args.ff_hidden_dim,
        num_heads=args.num_heads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dropout=args.dropout,
        max_position_embeddings=max(args.max_src_len, args.max_tgt_len),
        pad_token_id=pad_id,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        tie_embeddings=True,
    )
    model = Seq2Seq(config).to(device)

    # Optional torch.compile for speedup (PyTorch 2.0+)
    if args.compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Optimizer & schedule (warmup + inverse sqrt)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))

    # Mixed precision setup (bf16 on 4090 for best perf)
    use_amp = args.use_amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    if use_amp:
        print(f"Mixed precision enabled with dtype={amp_dtype}")

    # Optional resume
    start_step = 0
    resume_path = None
    if args.resume_from:
        resume_path = args.resume_from
    elif args.resume:
        resume_path = os.path.join(args.out_dir, "latest.pt")
    if resume_path:
        assert os.path.exists(resume_path), f"Resume checkpoint not found: {resume_path}"
        ckpt = torch.load(resume_path, map_location="cpu")
        assert "config" in ckpt and "model" in ckpt, "Resume checkpoint missing keys 'config'/'model'"
        # Rebuild model from checkpoint config to ensure exact shape match
        config = Seq2SeqConfig(**ckpt["config"])
        model = Seq2Seq(config).to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        start_step = 0 if args.reset_steps else int(ckpt.get("step", 0))
        # Load optimizer if available and not reset
        if (not args.reset_optim) and ("optimizer" in ckpt):
            try:
                opt.load_state_dict(ckpt["optimizer"])  # type: ignore[arg-type]
            except Exception:
                pass
        # Basic sanity warning for token ids
        if (pad_id != config.pad_token_id) or (bos_id != config.bos_token_id) or (eos_id != config.eos_token_id):
            print("[warn] Token IDs from tokenizer differ from checkpoint config; ensure you are using the same tokenizer.")
        print(f"Resumed from {resume_path} at step {start_step}.")

    def lr_factor(step: int):
        if step < max(1, args.warmup_steps):
            return step / max(1, args.warmup_steps)
        return (args.warmup_steps ** 0.5) / (step ** 0.5)

    def set_lr(step: int):
        for pg in opt.param_groups:
            pg["lr"] = args.lr * lr_factor(step)

    # Train loop
    model.train()
    total_steps = args.steps
    step = start_step
    start_time = time.time()
    avg_step_time = None
    best_val = float("inf")
    no_improve = 0

    # Prime ETA based on a couple of batches
    grad_accum_steps = max(1, args.grad_accum_steps)
    accum_loss = 0.0
    while step < total_steps:
        for src, src_m, dec_in, dec_m, labels in train_loader:
            step_start = time.time()
            # Only update LR at actual optimizer steps
            if (step % grad_accum_steps) == 0:
                set_lr((step // grad_accum_steps) + 1)

            src = src.to(device)
            src_m = src_m.to(device)
            dec_in = dec_in.to(device)
            dec_m = dec_m.to(device)
            labels = labels.to(device)

            # Forward pass with optional AMP
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                loss, _ = model(src, src_m, dec_in, dec_m, labels)
                loss = loss / grad_accum_steps  # Scale for accumulation

            # Backward pass
            if use_amp and amp_dtype == torch.float16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss += loss.item() * grad_accum_steps

            # Optimizer step (every grad_accum_steps)
            if (step + 1) % grad_accum_steps == 0:
                if args.grad_clip and args.grad_clip > 0:
                    if use_amp and amp_dtype == torch.float16:
                        scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if use_amp and amp_dtype == torch.float16:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

            step += 1
            step_time = time.time() - step_start
            avg_step_time = step_time if avg_step_time is None else 0.9 * avg_step_time + 0.1 * step_time

            if step % args.log_interval == 0 or step == 1:
                eta = (total_steps - step) * (avg_step_time or step_time)
                mins = int(eta // 60)
                secs = int(eta % 60)
                log_loss = accum_loss / min(step, args.log_interval)
                print(f"step {step}/{total_steps} | loss {log_loss:.4f} | lr {opt.param_groups[0]['lr']:.2e} | eta {mins:02d}:{secs:02d}")
                accum_loss = 0.0

            if args.eval_interval and step % args.eval_interval == 0:
                val_loss = evaluate(model, val_loader, device)
                print(f"eval step {step}: val_loss {val_loss:.4f}")
                # Print sample translations
                if args.print_samples and args.print_samples > 0:
                    previews = preview_samples(model, tok, val_ds, device, n=args.print_samples, max_new_tokens=args.max_tgt_len)
                    for j, (src_text, ref_text, hyp_text) in enumerate(previews):
                        print(f"sample {j+1} src: {src_text}")
                        print(f"sample {j+1} ref: {ref_text}")
                        print(f"sample {j+1} hyp: {hyp_text}")
                # Early stopping and best checkpoint
                if val_loss < best_val:
                    best_val = val_loss
                    no_improve = 0
                    save_ckpt(os.path.join(args.out_dir, "best.pt"), model, step, config, opt)
                else:
                    no_improve += 1
                    if args.patience_evals and no_improve >= args.patience_evals:
                        print(f"Early stopping: no improvement in {no_improve} evals. Best val {best_val:.4f}")
                        final_path = os.path.join(args.out_dir, "final.pt")
                        save_ckpt(final_path, model, step, config, opt)
                        elapsed = time.time() - start_time
                        print(f"Done. Steps: {step}, elapsed {elapsed/60:.1f} min. Saved to {final_path}")
                        return

            if args.save_interval and step % args.save_interval == 0:
                ckpt_path = os.path.join(args.out_dir, f"step_{step}.pt")
                save_ckpt(ckpt_path, model, step, config, opt)
                # also save latest
                save_ckpt(os.path.join(args.out_dir, "latest.pt"), model, step, config, opt)

            if step >= total_steps:
                break

    final_path = os.path.join(args.out_dir, "final.pt")
    save_ckpt(final_path, model, step, config, opt)
    elapsed = time.time() - start_time
    print(f"Done. Steps: {step}, elapsed {elapsed/60:.1f} min. Saved to {final_path}")


if __name__ == "__main__":
    main()
