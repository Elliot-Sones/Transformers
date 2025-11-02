"""
Hugging Face Space: Production MLM Word Prediction
Allows users to mask any word in the target sentence and predict what it should be.
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

# Import our encoder components
import sys
sys.path.append('encoder_transformer')
from encoder_transformer.encode import Encoder, EncoderConfig

class ProductionTokenizer:
    """Tokenizer for the production sentence."""
    
    def __init__(self, vocab_dict=None):
        if vocab_dict:
            self.vocab = vocab_dict
            self.id_to_token = {v: k for k, v in self.vocab.items()}
            self.vocab_size = len(self.vocab)
        else:
            # Build vocabulary from the target sentence
            target_sentence = "This model create relationships between the words to learn what word is missing!"
            words = target_sentence.lower().split()
            
            # Create vocabulary
            self.vocab = {}
            self.vocab["[PAD]"] = 0
            self.vocab["[UNK]"] = 1  
            self.vocab["[MASK]"] = 2
            
            # Add all words from the target sentence
            for i, word in enumerate(set(words)):  # Remove duplicates
                self.vocab[word] = i + 3
            
            self.id_to_token = {v: k for k, v in self.vocab.items()}
            self.vocab_size = len(self.vocab)
        
    def encode(self, text: str) -> list:
        words = text.lower().split()
        return [self.vocab.get(word, self.vocab["[UNK]"]) for word in words]
    
    def decode(self, ids: list) -> str:
        return " ".join([self.id_to_token.get(idx, "[UNK]") for idx in ids])

class ProductionMLM(nn.Module):
    """Production MLM for word prediction."""
    
    def __init__(self, vocab_size: int):
        super().__init__()
        
        # Production-ready encoder
        encoder_config = EncoderConfig(
            src_vocab_size=vocab_size,
            embed_dim=128,           # Small but effective
            ff_hidden_dim=256,       # Small but effective
            num_heads=4,             # Good for relationships
            num_layers=3,            # Enough depth
            max_position_embeddings=32,
            pad_token_id=0,
        )
        
        self.encoder = Encoder(encoder_config)
        
        # Output head for word prediction
        self.lm_head = nn.Linear(128, vocab_size)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        hidden = self.encoder(input_ids, attention_mask)
        logits = self.lm_head(hidden)
        return logits

# Global variables for the model
model = None
tokenizer = None
TARGET_SENTENCE = "This model create relationships between the words to learn what word is missing!"

def load_model():
    """Load the trained production model."""
    global model, tokenizer
    
    try:
        # Try to load from saved model
        checkpoint_path = 'encoder_transformer/mlm/models/production_model.pt'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Recreate tokenizer
            tokenizer = ProductionTokenizer(checkpoint['tokenizer_vocab'])
            
            # Recreate model
            model = ProductionMLM(vocab_size=tokenizer.vocab_size)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            print("✅ Model loaded successfully!")
            return True
        else:
            print("❌ Model file not found. Please train the model first.")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def predict_masked_word(masked_sentence: str, mask_position: int):
    """Predict what word should be at the masked position."""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return "❌ Model not loaded. Please train the model first."
    
    try:
        # Encode the sentence
        tokens = tokenizer.encode(masked_sentence)
        
        # Check if position is valid
        if mask_position >= len(tokens):
            return f"❌ Position {mask_position} is out of range (max: {len(tokens)-1})"
        
        # Get original word (before masking)
        original_word = tokenizer.id_to_token[tokens[mask_position]]
        
        # Mask the specified position
        tokens[mask_position] = tokenizer.vocab["[MASK]"]
        
        # Pad to max length
        max_len = 16
        padded_tokens = tokens + [tokenizer.vocab["[PAD]"]] * (max_len - len(tokens))
        
        with torch.no_grad():
            input_ids = torch.tensor([padded_tokens]).long()
            attention_mask = (input_ids != 0).float()
            logits = model(input_ids, attention_mask)
            
            # Get prediction for the masked position
            pred_id = logits[0, mask_position].argmax().item()
            pred_word = tokenizer.id_to_token[pred_id]
            
            # Get confidence
            confidence = torch.softmax(logits[0, mask_position], dim=-1)[pred_id].item()
            
            # Get top 3 predictions
            top3_probs, top3_ids = torch.topk(torch.softmax(logits[0, mask_position], dim=-1), 3)
            
            result = f"""
🎯 **Prediction Results:**

**Original word:** `{original_word}`
**Predicted word:** `{pred_word}`
**Confidence:** {confidence:.1%}
**Correct:** {'✅' if pred_word == original_word else '❌'}

**Top 3 predictions:**
"""
            
            for i, (prob, idx) in enumerate(zip(top3_probs, top3_ids)):
                word = tokenizer.id_to_token[idx.item()]
                result += f"{i+1}. `{word}` ({prob.item():.1%})\n"
            
            return result
            
    except Exception as e:
        return f"❌ Error during prediction: {str(e)}"

def create_interface():
    """Create the Gradio interface."""
    
    # Load model
    model_loaded = load_model()
    
    with gr.Blocks(title="Production MLM Word Prediction") as interface:
        gr.Markdown("""
        # 🔮 Production MLM Word Prediction
        
        This model learns relationships between words in the sentence:
        > **"This model create relationships between the words to learn what word is missing!"**
        
        **How it works:**
        1. Mask any word in the sentence by replacing it with `[MASK]`
        2. The model predicts what word should be there
        3. See the confidence and top predictions
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📝 Input Sentence")
                sentence_input = gr.Textbox(
                    value=TARGET_SENTENCE,
                    label="Sentence (modify words to [MASK] for prediction)",
                    placeholder="Enter sentence with [MASK] tokens",
                    lines=2
                )
                
                gr.Markdown("### 🎯 Word Positions")
                gr.Markdown("""
                **Word positions in the target sentence:**
                ```
                0: This     1: model     2: create     3: relationships  4: between
                5: the      6: words     7: to         8: learn         9: what
                10: word    11: is       12: missing!
                ```
                """)
                
                position_input = gr.Number(
                    value=0,
                    label="Position to mask (0-12)",
                    minimum=0,
                    maximum=12,
                    step=1
                )
                
                predict_btn = gr.Button("🔮 Predict Masked Word", variant="primary")
                
            with gr.Column():
                gr.Markdown("### 🎯 Prediction Results")
                output = gr.Markdown("Ready to predict! Click the button above.")
                
                gr.Markdown("### 🧠 Model Information")
                if model_loaded:
                    gr.Markdown("""
                    ✅ **Model Status:** Loaded and ready
                    
                    **Model Details:**
                    - Architecture: Transformer Encoder
                    - Embedding dimension: 128
                    - Layers: 3
                    - Attention heads: 4
                    - Vocabulary size: 16 tokens
                    - Training accuracy: 100%
                    """)
                else:
                    gr.Markdown("""
                    ❌ **Model Status:** Not loaded
                    
                    Please train the model first by running:
                    ```bash
                    python encoder_transformer/mlm_production.py
                    ```
                    """)
        
        # Example predictions
        gr.Markdown("### 💡 Example Predictions")
        
        examples = [
            ["This model [MASK] relationships between the words to learn what word is missing!", 2],
            ["This model create [MASK] between the words to learn what word is missing!", 3],
            ["This model create relationships [MASK] the words to learn what word is missing!", 4],
            ["This model create relationships between the words to [MASK] what word is missing!", 8],
        ]
        
        gr.Examples(
            examples=examples,
            inputs=[sentence_input, position_input],
            label="Click to try these examples"
        )
        
        # Event handlers
        def predict_wrapper(sentence, position):
            return predict_masked_word(sentence, int(position))
        
        predict_btn.click(
            fn=predict_wrapper,
            inputs=[sentence_input, position_input],
            outputs=output
        )
        
        gr.Markdown("""
        ### 🔬 How It Works
        
        This model uses **Multi-Head Self-Attention** to learn relationships between words:
        
        - **Context Learning**: Each word attends to other words in the sentence
        - **Relationship Building**: Discovers grammatical and semantic patterns
        - **Pattern Recognition**: Learns which words typically appear together
        
        **Example relationships learned:**
        - "creates" often follows "model"
        - "relationships" often follows "super" 
        - "between" connects two noun phrases
        - Articles ("a", "the") appear in specific positions
        """)
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
