"""
🤖 Transformer Learning Journey - Unified Hugging Face Spaces App
Educational implementation showcasing the "Attention Is All You Need" architecture progression.

Three tabs demonstrating:
1. Encoder-only (Emotion Classification)
2. Decoder-only (Shakespeare Generation)
3. Full Seq2Seq (EN→FR Translation)
"""

import os
import sys
from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Tokenizers library
try:
    from tokenizers import Tokenizer
except ImportError:
    print("Please install: pip install tokenizers")
    sys.exit(1)

# ============================================================================
# DEVICE DETECTION
# ============================================================================

def detect_device() -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = detect_device()

# ============================================================================
# MODEL 1: EMOTION CLASSIFIER (Encoder-only)
# ============================================================================

# Lazy loading globals
EMOTION_MODEL = None
EMOTION_TOKENIZER = None
EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EMOTION_MAX_LEN = 48

def get_emotion_model():
    """Lazy load the emotion classifier model and tokenizer."""
    global EMOTION_MODEL, EMOTION_TOKENIZER
    
    if EMOTION_MODEL is not None:
        return EMOTION_MODEL, EMOTION_TOKENIZER
    
    # Import components
    from encoder_transformer.sentiment.train import EmotionClassifier
    
    # Checkpoint path
    ckpt_path = PROJECT_ROOT / "encoder_transformer" / "sentiment" / "checkpoints" / "model_optimized.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Emotion model checkpoint not found: {ckpt_path}")
    
    print(f"Loading Emotion model from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    
    # Load tokenizer from path in checkpoint, or from same directory
    tokenizer_path = ckpt.get("tokenizer_path", "")
    if not tokenizer_path or not os.path.exists(tokenizer_path):
        tokenizer_path = str(ckpt_path.parent / "tokenizer.json")
    
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Emotion tokenizer not found: {tokenizer_path}")
    
    EMOTION_TOKENIZER = Tokenizer.from_file(tokenizer_path)
    
    # Build model from config
    config = ckpt["config"]
    EMOTION_MODEL = EmotionClassifier(
        vocab_size=config["vocab_size"],
        num_classes=config["num_classes"],
        embed_dim=config["embed_dim"],
        ff_hidden_dim=config["ff_hidden_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_seq_len=config["max_seq_len"],
        dropout=config["dropout"],
        pool_method=config["pool_method"],
        pad_token_id=config["pad_token_id"],
    ).to(DEVICE)
    
    EMOTION_MODEL.load_state_dict(ckpt["model_state_dict"])
    EMOTION_MODEL.eval()
    
    print(f"✅ Emotion model loaded! Best accuracy: {ckpt.get('best_acc', 'N/A')}")
    return EMOTION_MODEL, EMOTION_TOKENIZER


@torch.no_grad()
def predict_emotion(text: str) -> str:
    """Predict emotion for given text."""
    if not text.strip():
        return "Please enter some text to analyze."
    
    try:
        model, tokenizer = get_emotion_model()
        pad_id = tokenizer.token_to_id("[PAD]") or 0
        
        # Tokenize input (model was trained with max_seq_len=48)
        encoded = tokenizer.encode(text)
        input_ids = encoded.ids[:EMOTION_MAX_LEN]  # Truncate if needed
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(input_ids)
        
        # Pad to consistent length if needed (model handles variable lengths)
        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
        attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.float, device=DEVICE)
        
        # Get prediction
        loss, logits = model(input_ids_tensor, attention_mask_tensor)
        probs = F.softmax(logits, dim=-1)[0]
        
        # Get predicted class and confidence
        predicted_idx = torch.argmax(probs).item()
        predicted_emotion = EMOTION_LABELS[predicted_idx]
        confidence = probs[predicted_idx].item()
        
        # Build result with all probabilities
        emoji_map = {
            "sadness": "😢",
            "joy": "😄",
            "love": "❤️",
            "anger": "😠",
            "fear": "😨",
            "surprise": "😲"
        }
        
        result = f"## {emoji_map.get(predicted_emotion, '🎭')} Predicted Emotion: **{predicted_emotion.upper()}**\n\n"
        result += f"**Confidence:** {confidence:.1%}\n\n"
        result += "### Probability Distribution:\n\n"
        
        # Sort by probability for display
        sorted_probs = sorted(
            [(EMOTION_LABELS[i], probs[i].item(), emoji_map.get(EMOTION_LABELS[i], "")) 
             for i in range(len(EMOTION_LABELS))],
            key=lambda x: x[1],
            reverse=True
        )
        
        for emotion, prob, emoji in sorted_probs:
            bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
            result += f"{emoji} **{emotion}**: {bar} {prob:.1%}\n\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ============================================================================
# MODEL 2: SHAKESPEARE GENERATOR (Decoder-only)
# ============================================================================

SHAKESPEARE_MODEL = None
SHAKESPEARE_CHARS = None
SHAKESPEARE_ENCODE = None
SHAKESPEARE_DECODE = None
SHAKESPEARE_BLOCK_SIZE = 256

def get_shakespeare_model():
    """Lazy load the Shakespeare GPT model."""
    global SHAKESPEARE_MODEL, SHAKESPEARE_CHARS, SHAKESPEARE_ENCODE, SHAKESPEARE_DECODE, SHAKESPEARE_BLOCK_SIZE
    
    if SHAKESPEARE_MODEL is not None:
        return SHAKESPEARE_MODEL, SHAKESPEARE_ENCODE, SHAKESPEARE_DECODE
    
    # Import GPT model from sample.py (includes model definition)
    from decoder_transformer.sample import GPTLanguageModel
    
    # Checkpoint path
    ckpt_path = PROJECT_ROOT / "decoder_transformer" / "checkpoints" / "model_optimized.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Shakespeare model checkpoint not found: {ckpt_path}")
    
    print(f"Loading Shakespeare model from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    meta = ckpt["meta"]
    
    # Extract character vocabulary from checkpoint
    SHAKESPEARE_CHARS = meta["chars"]
    vocab_size = meta["vocab_size"]
    SHAKESPEARE_BLOCK_SIZE = meta["block_size"]
    
    # Build lookup tables
    lookup_in = {ch: i for i, ch in enumerate(SHAKESPEARE_CHARS)}
    lookup_out = {i: ch for i, ch in enumerate(SHAKESPEARE_CHARS)}
    
    SHAKESPEARE_ENCODE = lambda s: [lookup_in[c] for c in s if c in lookup_in]
    SHAKESPEARE_DECODE = lambda l: "".join([lookup_out[i] for i in l])
    
    # Build model
    SHAKESPEARE_MODEL = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=meta["n_embd"],
        n_head=meta["n_head"],
        n_layer=meta["n_layer"],
        block_size=SHAKESPEARE_BLOCK_SIZE,
        dropout=meta["dropout"],
    ).to(DEVICE)
    
    # Load weights (prefer EMA if available)
    state_key = "ema_state_dict" if "ema_state_dict" in ckpt and ckpt["ema_state_dict"] else "model_state_dict"
    state_dict = ckpt[state_key]
    
    # Strip torch.compile prefix if present
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    
    SHAKESPEARE_MODEL.load_state_dict(state_dict, strict=False)
    SHAKESPEARE_MODEL.eval()
    
    print(f"✅ Shakespeare model loaded! Vocab size: {vocab_size}")
    return SHAKESPEARE_MODEL, SHAKESPEARE_ENCODE, SHAKESPEARE_DECODE


@torch.no_grad()
def generate_shakespeare(prompt: str, max_tokens: int = 200) -> str:
    """Generate Shakespeare-style text."""
    if not prompt.strip():
        prompt = "ROMEO:"
    
    try:
        model, encode, decode = get_shakespeare_model()
        
        # Encode prompt
        start_tokens = encode(prompt)
        if len(start_tokens) == 0:
            start = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
        else:
            start = torch.tensor([start_tokens], dtype=torch.long, device=DEVICE)
        
        # Generate
        out = model.generate(start, max_new_tokens=int(max_tokens))
        text = decode(out[0].tolist())
        
        return f"## 📜 Generated Shakespeare:\n\n```\n{text}\n```"
        
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ============================================================================
# MODEL 3: EN→FR TRANSLATION (Encoder-Decoder Seq2Seq)
# ============================================================================

TRANSLATION_MODEL = None
TRANSLATION_TOKENIZER = None
TRANSLATION_MAX_SRC_LEN = 64
TRANSLATION_MAX_TGT_LEN = 64

def get_translation_model():
    """Lazy load the translation model and tokenizer."""
    global TRANSLATION_MODEL, TRANSLATION_TOKENIZER
    
    if TRANSLATION_MODEL is not None:
        return TRANSLATION_MODEL, TRANSLATION_TOKENIZER
    
    # Import components
    from machine_translation.mini_transformer import Seq2SeqConfig, Seq2Seq
    
    # Checkpoint path
    ckpt_path = PROJECT_ROOT / "machine_translation" / "checkpoints" / "model_optimized.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Translation model checkpoint not found: {ckpt_path}")
    
    # Tokenizer path - use the tokenizer_with_words.json
    tokenizer_path = PROJECT_ROOT / "machine_translation" / "tokenizer_with_words.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Translation tokenizer not found: {tokenizer_path}")
    
    print(f"Loading Translation model from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    
    # Load tokenizer
    TRANSLATION_TOKENIZER = Tokenizer.from_file(str(tokenizer_path))
    
    # Build model from config
    cfg = Seq2SeqConfig(**ckpt["config"])
    TRANSLATION_MODEL = Seq2Seq(cfg).to(DEVICE)
    TRANSLATION_MODEL.load_state_dict(ckpt["model"], strict=True)
    TRANSLATION_MODEL.eval()
    
    print(f"✅ Translation model loaded!")
    return TRANSLATION_MODEL, TRANSLATION_TOKENIZER


@torch.no_grad()
def translate_en_to_fr(text: str) -> str:
    """Translate English text to French."""
    if not text.strip():
        return "Please enter English text to translate."
    
    try:
        model, tokenizer = get_translation_model()
        pad_id = tokenizer.token_to_id("[PAD]") or 0
        
        # Tokenize source
        src_ids = tokenizer.encode(text).ids[:TRANSLATION_MAX_SRC_LEN]
        if not src_ids:
            src_ids = [pad_id]
        
        src = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)
        src_mask = (src != pad_id).long()
        
        # Generate translation
        ys = model.greedy_generate(
            src,
            src_mask,
            max_new_tokens=TRANSLATION_MAX_TGT_LEN,
            temperature=0.0,
        )
        ys = model.decode_tokens(ys)
        
        # Decode tokens
        row = [t for t in ys[0].tolist() if t != pad_id]
        french = tokenizer.decode(row)
        
        return f"## 🇫🇷 French Translation:\n\n**{french}**"
        
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_interface():
    """Create the unified Gradio interface with three tabs."""
    
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="purple",
    )
    
    with gr.Blocks(
        title="🤖 Transformer Learning Journey",
        theme=theme,
        css="""
        .header-text { text-align: center; margin-bottom: 20px; }
        .tab-content { padding: 20px; }
        """
    ) as interface:
        
        gr.Markdown("""
        # 🤖 Transformer Learning Journey
        
        > **Educational implementation of "Attention Is All You Need"**
        
        This app demonstrates the three major Transformer architectures:
        

        """, elem_classes="header-text")
        
        with gr.Tabs():
            # =================================================================
            # TAB 1: EMOTION CLASSIFICATION
            # =================================================================
            with gr.Tab("🎭 Emotion (Encoder)"):
                gr.Markdown("""
                ## Encoder-Only Architecture
                
                This model uses only the **Encoder** portion of the Transformer to classify emotions.
                The encoder processes the entire input to understand context, then a classification head predicts the emotion.
                
                **Labels:** sadness, joy, love, anger, fear, surprise
                """)
                
                with gr.Row():
                    with gr.Column():
                        emotion_input = gr.Textbox(
                            label="Enter text to analyze",
                            placeholder="I am so happy today! Everything is going great!",
                            lines=3,
                        )
                        emotion_btn = gr.Button("🎭 Classify Emotion", variant="primary")
                        
                        gr.Examples(
                            examples=[
                                ["I am so happy today!"],
                                ["I feel so sad and lonely."],
                                ["I love you with all my heart."],
                                ["This makes me so angry!"],
                                ["I'm scared of what might happen."],
                                ["Wow, I didn't expect that at all!"],
                            ],
                            inputs=emotion_input,
                        )
                    
                    with gr.Column():
                        emotion_output = gr.Markdown(
                            value="Enter text and click 'Classify Emotion' to see predictions."
                        )
                
                emotion_btn.click(predict_emotion, inputs=emotion_input, outputs=emotion_output)
            
            # =================================================================
            # TAB 2: SHAKESPEARE GENERATOR
            # =================================================================
            with gr.Tab("📜 Shakespeare (Decoder)"):
                gr.Markdown("""
                ## Decoder-Only Architecture
                
                This model uses only the **Decoder** portion of the Transformer (like GPT).
                It generates text character-by-character using causal (autoregressive) attention.
                
                **Trained on:** Complete works of Shakespeare
                """)
                
                with gr.Row():
                    with gr.Column():
                        shakespeare_prompt = gr.Textbox(
                            label="Starting prompt",
                            placeholder="ROMEO:",
                            value="ROMEO:",
                            lines=2,
                        )
                        shakespeare_tokens = gr.Slider(
                            minimum=50,
                            maximum=500,
                            value=200,
                            step=50,
                            label="Max tokens to generate",
                        )
                        shakespeare_btn = gr.Button("📜 Generate", variant="primary")
                        
                        gr.Examples(
                            examples=[
                                ["ROMEO:"],
                                ["JULIET:"],
                                ["To be, or not to be,"],
                                ["KING:"],
                                ["All the world's a stage,"],
                            ],
                            inputs=shakespeare_prompt,
                        )
                    
                    with gr.Column():
                        shakespeare_output = gr.Markdown(
                            value="Enter a prompt and click 'Generate' to create Shakespeare-style text."
                        )
                
                shakespeare_btn.click(
                    generate_shakespeare,
                    inputs=[shakespeare_prompt, shakespeare_tokens],
                    outputs=shakespeare_output,
                )
            
            # =================================================================
            # TAB 3: EN→FR TRANSLATION
            # =================================================================
            with gr.Tab("🇫🇷 EN→FR Translation (Full Seq2Seq)"):
                gr.Markdown("""
                ## Full Encoder-Decoder Architecture
                
                This model uses the complete **Encoder-Decoder** Transformer for sequence-to-sequence translation.
                The encoder processes the English input, and the decoder generates the French output.
                
                **Task:** English → French translation
                """)
                
                with gr.Row():
                    with gr.Column():
                        translation_input = gr.Textbox(
                            label="English text",
                            placeholder="Hello, how are you?",
                            lines=3,
                        )
                        translation_btn = gr.Button("🇫🇷 Translate", variant="primary")
                        
                        gr.Examples(
                            examples=[
                                ["Hello, how are you?"],
                                ["I love learning new languages."],
                                ["The weather is beautiful today."],
                                ["Thank you very much."],
                                ["Good morning!"],
                            ],
                            inputs=translation_input,
                        )
                    
                    with gr.Column():
                        translation_output = gr.Markdown(
                            value="Enter English text and click 'Translate' to see the French translation."
                        )
                
                translation_btn.click(
                    translate_en_to_fr,
                    inputs=translation_input,
                    outputs=translation_output,
                )
        
        gr.Markdown("""
        ---
        
        ### 🔬 About This Project
        
        This demo showcases three different Transformer architectures:
        
        1. **Encoder-only** (like BERT): Best for understanding/classification tasks
        2. **Decoder-only** (like GPT): Best for text generation tasks
        3. **Encoder-Decoder** (original Transformer): Best for sequence-to-sequence tasks
        
        All models are trained from scratch as educational implementations.
        """)
    
    return interface


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
    )
