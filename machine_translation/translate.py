import os
import argparse
from typing import List

import torch
from tokenizers import Tokenizer

try:
    from machine_translation.mini_transformer import Seq2SeqConfig, Seq2Seq
except ModuleNotFoundError:
    from mini_transformer import Seq2SeqConfig, Seq2Seq


def detect_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(ckpt_path: str) -> dict:
    assert os.path.exists(ckpt_path), f"Missing checkpoint: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert "config" in ckpt and "model" in ckpt, "Checkpoint missing keys 'config'/'model'"
    cfg = Seq2SeqConfig(**ckpt["config"])  # restore exactly as trained
    model = Seq2Seq(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return {"model": model, "config": cfg, "step": ckpt.get("step", 0)}


def batch_translate(model: Seq2Seq, tok: Tokenizer, texts: List[str], max_src_len: int, max_new_tokens: int,
                    temperature: float = 0.0, top_k: int = 0, device: torch.device = torch.device("cpu")) -> List[str]:
    pad_id = tok.token_to_id("[PAD]") or 0
    # Tokenize and pad to batch tensor
    src_ids_list = []
    for t in texts:
        ids = tok.encode(t).ids[:max_src_len]
        if not ids:
            ids = [pad_id]
        src_ids_list.append(ids)

    max_len = max(len(x) for x in src_ids_list)
    B = len(src_ids_list)
    src = torch.full((B, max_len), pad_id, dtype=torch.long)
    for i, ids in enumerate(src_ids_list):
        src[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    src_mask = (src != pad_id).long()

    src = src.to(device)
    src_mask = src_mask.to(device)

    with torch.no_grad():
        ys = model.greedy_generate(
            src,
            src_mask,
            max_new_tokens=max_new_tokens,
            temperature=float(temperature) if temperature else 0.0,
            top_k=int(top_k) if top_k else None,
        )
        ys = model.decode_tokens(ys)
        # Remove PAD for decoding
        outs = []
        for row in ys.tolist():
            row = [t for t in row if t != pad_id]
            outs.append(tok.decode(row))
        return outs


def build_argparser():
    p = argparse.ArgumentParser(description="Translate text with a trained mini Transformer")
    p.add_argument("--ckpt", type=str, default=os.path.join("machine_translation", "checkpoints", "mini", "best.pt"))
    p.add_argument("--tokenizer_path", type=str, default=os.path.join("machine_translation", "archive", "tokenizer.json"))
    p.add_argument("--text", type=str, default=None, help="Translate a single quoted sentence")
    p.add_argument("--file", type=str, default=None, help="Optional path to a text file; one sentence per line")
    p.add_argument("--max_src_len", type=int, default=64)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--batch", type=int, default=16, help="Batch size for file mode")
    return p


def main():
    args = build_argparser().parse_args()
    device = detect_device()
    print(f"Using device: {device}")

    bundle = load_model(args.ckpt)
    model: Seq2Seq = bundle["model"].to(device)
    tok = Tokenizer.from_file(args.tokenizer_path)

    if args.text:
        outs = batch_translate(model, tok, [args.text], args.max_src_len, args.max_new_tokens, args.temperature, args.top_k, device)
        print(outs[0])
        return

    if args.file:
        assert os.path.exists(args.file), f"Missing file: {args.file}"
        with open(args.file, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f]
        for i in range(0, len(lines), args.batch):
            batch = lines[i : i + args.batch]
            outs = batch_translate(model, tok, batch, args.max_src_len, args.max_new_tokens, args.temperature, args.top_k, device)
            for out in outs:
                print(out)
        return

    # Interactive mode
    print("Enter text to translate (Ctrl-D to quit):")
    try:
        while True:
            inp = input(">> ").strip()
            if not inp:
                print("")
                continue
            outs = batch_translate(model, tok, [inp], args.max_src_len, args.max_new_tokens, args.temperature, args.top_k, device)
            print(outs[0])
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    main()

