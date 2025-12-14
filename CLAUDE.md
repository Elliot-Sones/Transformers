# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational implementation of the Transformer architecture from "Attention Is All You Need" (2017), demonstrating encoder, decoder, and full sequence-to-sequence translation models.

## Commands

### Install Dependencies
```bash
pip install torch torchvision torchaudio datasets pandas tqdm tokenizers gradio
```

### Train Encoder (Masked Language Model)
```bash
python encoder_transformer/mlm/train.py
```

### Train Encoder (Emotion Classification) - CUDA Ready
```bash
python encoder_transformer/sentiment/train.py
python encoder_transformer/sentiment/train.py --epochs 5 --batch_size 64
python encoder_transformer/sentiment/train.py --resume --ckpt checkpoints/best.pt
```
- Uses `dair-ai/emotion` dataset (auto-downloaded)
- 6 classes: sadness, joy, love, anger, fear, surprise
- Mixed precision (FP16) on CUDA for faster training
- Checkpoints saved to `encoder_transformer/sentiment/checkpoints/`

### Predict Emotion
```bash
python encoder_transformer/sentiment/predict.py --text "I am so happy today!"
python encoder_transformer/sentiment/predict.py --interactive
python encoder_transformer/sentiment/predict.py --input_file texts.txt --output_file predictions.csv
```

### Train Decoder (GPT-style Language Model) - CUDA Ready
```bash
cd decoder_transformer && python training.py
```
- Auto-downloads Tiny Shakespeare dataset on first run
- Auto-detects CUDA → MPS → CPU
- Mixed precision (FP16) on CUDA/MPS for faster training
- Resume training: `python training.py --resume`
- Custom checkpoint: `python training.py --resume --ckpt assets/checkpoints/latest.pt`

### Generate Shakespeare-style Text
```bash
cd decoder_transformer && python sample.py --prompt "ROMEO:" --max_new_tokens 300
python sample.py --prompt "" --max_new_tokens 500  # Empty prompt = generate from scratch
python sample.py --device cuda --prompt "To be or not to be" --max_new_tokens 200
```

### Train Machine Translation (Full Seq2Seq)
```bash
# Prepare data
python machine_translation/setup_data.py --dataset wmt14 --out_dir data/en_fr

# Train
python machine_translation/train_mini.py --train_csv data/en_fr/train.csv --val_csv data/en_fr/test.csv
```
- Resume: `python train_mini.py --resume` or `--resume_from path/to/checkpoint.pt`

### Launch Web Interface
```bash
python app.py
```
Launches Gradio interface on port 7860 for MLM word prediction demo.

## Architecture

### Core Components

**encoder_transformer/encode.py**: Standalone encoder with:
- `EncoderConfig`: Dataclass for model hyperparameters
- `TokenPositionalEmbedding`: Token + learned positional embeddings
- `MultiHeadSelfAttention`: Self-attention with padding mask support
- `EncoderBlock`: Pre-LN transformer block (LN → MHA → residual → LN → FFN → residual)
- `Encoder`: Full encoder stack with final LayerNorm

**machine_translation/mini_transformer.py**: Full seq2seq implementation with:
- `Seq2SeqConfig`: Combined encoder/decoder config
- `MultiHeadAttention`: Flexible MHA supporting both self-attention and cross-attention via `kv` parameter
- `DecoderBlock`: Masked self-attention → cross-attention → FFN
- `Seq2Seq`: Complete encoder-decoder with greedy generation

**decoder_transformer/training.py**: GPT-style decoder-only model using character-level tokenization on Tiny Shakespeare.

### Key Patterns

- **Pre-LN architecture**: LayerNorm applied before attention/FFN (not after)
- **Attention masks**: `1/True` = keep token, `0/False` = pad (converted to bool internally)
- **Weight tying**: LM head shares weights with token embeddings
- **Causal masking**: Decoder self-attention uses upper-triangular mask for autoregressive generation
- **Device detection**: Auto-selects CUDA → MPS (Apple Silicon) → CPU

### Data Flow

1. Input tokens → Token embedding + Positional embedding
2. Encoder: N × (Self-Attention → FFN) with residuals
3. Decoder: N × (Masked Self-Attention → Cross-Attention → FFN)
4. LM head projects to vocabulary logits

## Project Structure

```
encoder_transformer/
├── encode.py              # Encoder architecture
├── mlm/                   # Masked language model training
│   ├── train.py
│   ├── test.py
│   └── models/            # Saved checkpoints
└── sentiment/             # Emotion classification (CUDA ready)
    ├── train.py           # Training script (6-way emotion)
    ├── predict.py         # Inference script
    └── checkpoints/       # Saved models

decoder_transformer/
├── training.py            # GPT training script
└── sample.py              # Text generation

machine_translation/
├── mini_transformer.py    # Full seq2seq model
├── train_mini.py          # Training script
├── setup_data.py          # Data preparation
└── translate.py           # Inference

app.py                     # Gradio web interface
```
