"""
Setup script to download and prepare EN→FR machine translation data.

This script:
 1) Downloads a large parallel dataset (WMT14, WMT16, or OPUS Books) via
    Hugging Face datasets (automatic URL handling under the hood).
 2) Normalizes text, filters extremes (length and length-ratio), and
    optionally de-duplicates pairs.
 3) Creates deterministic train/val/test splits if missing, or preserves
    provided splits.
 4) Saves cleaned CSVs with columns: en, fr.

Usage (examples):
  python -m machine_translation.setup_data --dataset wmt14 --out_dir data/en_fr/wmt14
  python -m machine_translation.setup_data --dataset opus_books --max_src_chars 256 --max_tgt_chars 256 \
      --train_ratio 0.9 --val_ratio 0.05 --test_ratio 0.05

Notes:
 - Requires: datasets, pandas, tqdm (install via: pip install datasets pandas tqdm)
 - By default, this script keeps a lot of data ("heavy side") while applying
   sensible filters to remove clearly problematic examples.
"""

import argparse
import json
import os
import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Tuple


def _require(module_name: str, pip_hint: str) -> object:
    """Import a module or raise a helpful error with pip instructions."""
    try:
        return __import__(module_name)
    except Exception as e:
        raise RuntimeError(
            f"Missing dependency: {module_name}. Install with: pip install {pip_hint}\nOriginal error: {e}"
        )


datasets = _require("datasets", "datasets")
pd = _require("pandas", "pandas")
tqdm = _require("tqdm", "tqdm").tqdm


DATASET_CATALOG: Dict[str, Tuple[str, str]] = {
    # dataset_name -> (hf_dataset_id, hf_config)
    # WMT14 is a strong, large EN–FR corpus. Config orientation is fr-en.
    "wmt14": ("wmt14", "fr-en"),
    # WMT16 also offers fr-en; sometimes slightly different domain mix.
    "wmt16": ("wmt16", "fr-en"),
    # OPUS Books is smaller than WMT but still sizeable and cleanly aligned.
    "opus_books": ("opus_books", "en-fr"),
}


def normalize_text(s: str, lowercase: bool = False) -> str:
    """Unicode-normalize and lightly clean whitespace/punctuation spacing.

    - NFKC normalization stabilizes different Unicode presentations.
    - Collapses excessive whitespace, strips ends.
    - Optional lowercasing is provided but off by default for case-sensitive tokenizers.
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u200b", "")  # zero-width space
    s = re.sub(r"\s+", " ", s).strip()
    if lowercase:
        s = s.lower()
    return s


def count_words(s: str) -> int:
    return 0 if not s else len(s.split())


def length_metrics(en: str, fr: str, metric: str) -> Tuple[int, int]:
    if metric == "words":
        return count_words(en), count_words(fr)
    return len(en), len(fr)  # chars


def ratio_ok(en_len: int, fr_len: int, min_ratio: float, max_ratio: float) -> bool:
    # Allow zero lengths to be filtered out by other checks.
    if en_len == 0 or fr_len == 0:
        return False
    r = max(en_len, fr_len) / max(1, min(en_len, fr_len))
    return (r >= min_ratio) and (r <= max_ratio)


def load_dataset(dataset_name: str,
                 server_slice_train: Optional[str] = None,
                 server_slice_eval: Optional[str] = None):
    """Load a dataset; optionally use server-side slicing to avoid full download.

    server_slice_* accept values like "1000000" or "10%" and will be applied as
    split strings: e.g., train[:1000000], validation[:10000], test[:10000].
    """
    if dataset_name not in DATASET_CATALOG:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(DATASET_CATALOG)}")
    hf_id, hf_config = DATASET_CATALOG[dataset_name]

    # If no slicing requested, load the full DatasetDict as before
    if not server_slice_train and not server_slice_eval:
        try:
            return datasets.load_dataset(hf_id, hf_config)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset {hf_id} with config {hf_config}.\n"
                f"Ensure network access and that the dataset is available. Original error: {e}"
            )

    # Server-side sliced loads: build a dict of available splits
    out = {}
    # Train
    if server_slice_train:
        try:
            out["train"] = datasets.load_dataset(hf_id, hf_config, split=f"train[:{server_slice_train}]")
        except Exception:
            pass
    # Evaluation: try validation, valid, dev, test
    for name in ["validation", "valid", "dev", "test"]:
        if server_slice_eval:
            try:
                out[name] = datasets.load_dataset(hf_id, hf_config, split=f"{name}[:{server_slice_eval}]")
            except Exception:
                continue
    if not out:
        # Fall back to full load if slicing failed
        try:
            return datasets.load_dataset(hf_id, hf_config)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset {hf_id} with config {hf_config}.\n"
                f"Ensure network access and that the dataset is available. Original error: {e}"
            )
    return out


def extract_en_fr(example: Dict) -> Optional[Tuple[str, str]]:
    """Extract (en, fr) from a datasets 'translation' example regardless of key order."""
    tr = example.get("translation")
    if not isinstance(tr, dict):
        return None
    # Normalize any key casing just in case, though HF is consistent.
    keys = {k.lower(): k for k in tr.keys()}
    k_en = keys.get("en")
    k_fr = keys.get("fr")
    if k_en is None or k_fr is None:
        return None
    return tr.get(k_en, None), tr.get(k_fr, None)


def dataset_to_pairs(ds_split, lowercase: bool, max_src_chars: int, max_tgt_chars: int,
                     max_src_words: Optional[int], max_tgt_words: Optional[int],
                     ratio_metric: str, min_ratio: float, max_ratio: float,
                     limit: Optional[int] = None) -> List[Tuple[str, str]]:
    """Convert a HF datasets split to a cleaned list of (en, fr) pairs.

    This performs normalization and filtering but does not de-duplicate.
    """
    pairs: List[Tuple[str, str]] = []
    removed_empty = 0
    removed_length = 0
    removed_ratio = 0

    iterable = ds_split
    if limit is not None:
        iterable = ds_split.select(range(min(limit, len(ds_split))))

    for ex in tqdm(iterable, desc="cleaning", total=(len(iterable) if hasattr(iterable, "__len__") else None)):
        row = extract_en_fr(ex)
        if row is None:
            removed_empty += 1
            continue
        en_raw, fr_raw = row
        en = normalize_text(en_raw, lowercase=lowercase)
        fr = normalize_text(fr_raw, lowercase=lowercase)

        if not en or not fr:
            removed_empty += 1
            continue

        # Length filters (chars)
        if len(en) > max_src_chars or len(fr) > max_tgt_chars:
            removed_length += 1
            continue

        # Optional length filters (words)
        if max_src_words is not None and count_words(en) > max_src_words:
            removed_length += 1
            continue
        if max_tgt_words is not None and count_words(fr) > max_tgt_words:
            removed_length += 1
            continue

        # Ratio filter
        e_len, f_len = length_metrics(en, fr, metric=ratio_metric)
        if not ratio_ok(e_len, f_len, min_ratio=min_ratio, max_ratio=max_ratio):
            removed_ratio += 1
            continue

        pairs.append((en, fr))

    print(
        f"Removed: empty={removed_empty}, length={removed_length}, ratio={removed_ratio}. "
        f"Kept={len(pairs)}"
    )
    return pairs


def deduplicate_pairs(pairs: List[Tuple[str, str]], max_items_hint: Optional[int] = None) -> List[Tuple[str, str]]:
    """Optional in-memory deduplication. For very large corpora, this may be memory-heavy.
    max_items_hint is only used for logging; dedup proceeds regardless.
    """
    seen = set()
    out = []
    for en, fr in tqdm(pairs, desc="dedup"):
        key = (en, fr)
        if key in seen:
            continue
        seen.add(key)
        out.append((en, fr))
    if max_items_hint is not None:
        print(f"Dedup done. Unique={len(out)} of ~{max_items_hint} input")
    else:
        print(f"Dedup done. Unique={len(out)}")
    return out


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def to_dataframe(pairs: List[Tuple[str, str]]):
    return pd.DataFrame({"en": [p[0] for p in pairs], "fr": [p[1] for p in pairs]})


def deterministic_split(df, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Ratios must sum to 1.0"
    # Shuffle deterministically by a stable hash of content + seed
    rng = pd.util.hash_pandas_object(df["en"] + "\u0001" + df["fr"], index=False) ^ seed
    df = df.assign(_h=rng).sort_values("_h").drop(columns=["_h"]).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]
    return train_df, val_df, test_df


def _noop(*_args, **_kwargs):
    return None


def prepare(args: argparse.Namespace):
    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(
        args.dataset,
        server_slice_train=args.server_slice_train,
        server_slice_eval=args.server_slice_eval,
    )

    # Consolidate pairs from available splits (heavy side: keep all available)
    raw_splits = []
    for split_name in ["train", "validation", "valid", "dev", "test"]:
        if split_name in ds:
            raw_splits.append(split_name)

    print(f"Found splits: {raw_splits if raw_splits else '[none]'}")

    all_pairs: List[Tuple[str, str]] = []
    for sn in raw_splits or ["train"]:
        if sn not in ds:
            continue
        print(f"Processing split '{sn}' (size={len(ds[sn])})")
        pairs = dataset_to_pairs(
            ds[sn],
            lowercase=args.lowercase,
            max_src_chars=args.max_src_chars,
            max_tgt_chars=args.max_tgt_chars,
            max_src_words=args.max_src_words,
            max_tgt_words=args.max_tgt_words,
            ratio_metric=args.ratio_metric,
            min_ratio=args.min_ratio,
            max_ratio=args.max_ratio,
            limit=args.limit,
        )
        all_pairs.extend(pairs)

    before_dedup = len(all_pairs)
    if args.dedup:
        all_pairs = deduplicate_pairs(all_pairs, max_items_hint=before_dedup)

    kept_total = len(all_pairs)
    print(f"Total kept examples after cleaning{' and dedup' if args.dedup else ''}: {kept_total}")
    if kept_total == 0:
        raise RuntimeError("No examples left after cleaning. Relax filters or check dataset.")

    # Global cap on total examples (default 1,000,000) to keep dataset manageable
    if args.max_examples_total is not None and kept_total > args.max_examples_total:
        all_pairs = all_pairs[:args.max_examples_total]
        kept_total = len(all_pairs)
        print(f"Capped total examples to {kept_total} (max_examples_total={args.max_examples_total})")

    out_dir = ensure_dir(args.out_dir)
    # Save merged then split deterministically unless dataset already had distinct splits and user wants preserve.
    merged_df = to_dataframe(all_pairs)

    if args.preserve_original_splits and set(raw_splits) >= {"train", "validation", "test"}:
        # Re-clean per split and then write only train.csv and test.csv (validation merged into test).
        cleaned = {}
        for sn in ["train", "validation", "test"]:
            if sn in ds:
                pairs = dataset_to_pairs(
                    ds[sn],
                    lowercase=args.lowercase,
                    max_src_chars=args.max_src_chars,
                    max_tgt_chars=args.max_tgt_chars,
                    max_src_words=args.max_src_words,
                    max_tgt_words=args.max_tgt_words,
                    ratio_metric=args.ratio_metric,
                    min_ratio=args.min_ratio,
                    max_ratio=args.max_ratio,
                    limit=args.limit,
                )
                if args.dedup:
                    pairs = deduplicate_pairs(pairs)
                cleaned[sn] = to_dataframe(pairs)

        train_df = cleaned.get("train", pd.DataFrame({"en": [], "fr": []}))
        # Merge validation and test into one test set
        test_parts = []
        if "validation" in cleaned:
            test_parts.append(cleaned["validation"])
        if "test" in cleaned:
            test_parts.append(cleaned["test"])
        test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame({"en": [], "fr": []})

        train_csv = os.path.join(out_dir, "train.csv")
        test_csv = os.path.join(out_dir, "test.csv")
        train_df.to_csv(train_csv, index=False)
        test_df.to_csv(test_csv, index=False)
        print(f"Wrote train/test -> {train_csv} ({len(train_df)}), {test_csv} ({len(test_df)})")

        _noop()
        return

    # Deterministic re-split
    train_df, val_df, test_df = deterministic_split(
        merged_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # Always emit only train.csv and test.csv: merge validation into test.
    test_df_merged = pd.concat([val_df, test_df], ignore_index=True)

    train_csv = os.path.join(out_dir, "train.csv")
    test_csv = os.path.join(out_dir, "test.csv")
    train_df.to_csv(train_csv, index=False)
    test_df_merged.to_csv(test_csv, index=False)

    print(f"Wrote train/test -> {train_csv} ({len(train_df)}), {test_csv} ({len(test_df_merged)})")

    _noop()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download and prepare EN→FR MT data")
    p.add_argument("--dataset", type=str, default="wmt14", choices=list(DATASET_CATALOG.keys()),
                   help="Which dataset to download and prepare")
    p.add_argument("--out_dir", type=str, default=os.path.join("machine_translation", "archive"),
                   help="Output directory (default: machine_translation/archive)")

    # Server-side slicing to avoid downloading the full corpus
    p.add_argument("--server_slice_train", type=str, default="1000000",
                   help="Server-side slice for train split (e.g., '1000000' or '10%'). None downloads full.")
    p.add_argument("--server_slice_eval", type=str, default="10000",
                   help="Server-side slice for eval splits (validation/test). None downloads full.")

    # Cleaning and filtering options
    p.add_argument("--lowercase", action="store_true", help="Lowercase all text during normalization")
    p.add_argument("--max_src_chars", type=int, default=192, help="Max source length in characters (shorter for production)")
    p.add_argument("--max_tgt_chars", type=int, default=192, help="Max target length in characters (shorter for production)")
    p.add_argument("--max_src_words", type=int, default=None, help="Optional max source length in words")
    p.add_argument("--max_tgt_words", type=int, default=None, help="Optional max target length in words")
    p.add_argument("--ratio_metric", type=str, default="words", choices=["words", "chars"],
                   help="Use word or char counts for length-ratio filtering")
    p.add_argument("--min_ratio", type=float, default=0.5, help="Min acceptable max/min length ratio")
    p.add_argument("--max_ratio", type=float, default=2.0, help="Max acceptable max/min length ratio")
    p.add_argument("--dedup", action="store_true", help="Enable in-memory deduplication of (en, fr) pairs")
    p.add_argument("--limit", type=int, default=None, help="Limit examples per split (debugging)")
    p.add_argument("--max_examples_total", type=int, default=500000, help="Cap total cleaned examples (500k for faster training)")

    # Split options
    p.add_argument("--train_ratio", type=float, default=0.98, help="Train ratio when re-splitting (heavy train)")
    p.add_argument("--val_ratio", type=float, default=0.01, help="Validation ratio when re-splitting")
    p.add_argument("--test_ratio", type=float, default=0.01, help="Test ratio when re-splitting")
    p.add_argument("--seed", type=int, default=1337, help="Deterministic shuffling seed for re-split")
    p.add_argument("--preserve_original_splits", action="store_true",
                   help="If dataset has train/validation/test, save each cleaned split separately instead of re-splitting")
    return p


def main():
    args = build_argparser().parse_args()
    prepare(args)


if __name__ == "__main__":
    main()


