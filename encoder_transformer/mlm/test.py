"""
Test the trained encoder MLM model.
"""

import torch
import torch.nn as nn
from train import ProductionMLM, ProductionTokenizer

def load_model():
    """Load the trained model."""
    try:
        checkpoint = torch.load('models/production_model.pt', map_location='cpu')
        
        # Recreate tokenizer
        tokenizer = ProductionTokenizer()
        
        # Recreate model
        model = ProductionMLM(vocab_size=tokenizer.vocab_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def predict_masked_word(model, tokenizer, sentence: str, mask_position: int):
    """Predict what word should be at the masked position."""
    # Encode the sentence
    tokens = tokenizer.encode(sentence)
    
    # Mask the specified position
    if mask_position >= len(tokens):
        return "Error: Position out of range"
    
    original_word = tokenizer.id_to_token[tokens[mask_position]]
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
        
        return {
            'original_word': original_word,
            'predicted_word': pred_word,
            'correct': pred_word == original_word,
            'confidence': confidence
        }

def test_all_positions():
    """Test masking each word position in the target sentence."""
    print("Loading trained MLM model...")
    model, tokenizer = load_model()
    
    if model is None:
        print("❌ Could not load model. Please train first with: python train.py")
        return
    
    target_sentence = "This model create relationships between the words to learn what word is missing!"
    words = target_sentence.split()
    
    print(f"\nTesting word prediction on: '{target_sentence}'")
    print("="*80)
    
    correct_predictions = 0
    total_predictions = len(words)
    
    for i, word in enumerate(words):
        result = predict_masked_word(model, tokenizer, target_sentence, i)
        
        status = "✓" if result['correct'] else "✗"
        confidence = result['confidence']
        
        print(f"Position {i:2d}: '{word}' → '{result['predicted_word']}' {status} (conf: {confidence:.3f})")
        
        if result['correct']:
            correct_predictions += 1
    
    print("="*80)
    print(f"Overall accuracy: {correct_predictions}/{total_predictions} = {correct_predictions/total_predictions*100:.1f}%")
    
    return model, tokenizer

if __name__ == "__main__":
    test_all_positions()
