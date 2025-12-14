"""
Data preparation for encoder MLM training.
"""

import os
import pandas as pd

def create_mlm_data():
    """Create training data for the MLM model."""
    
    # Target sentence for production
    target_sentence = "This model create relationships between the words to learn what word is missing!"
    
    # Create variations by repeating the sentence many times for training
    train_data = []
    for _ in range(1000):  # 1000 training samples
        train_data.append(target_sentence)
    
    # Create smaller validation and test sets
    val_data = []
    for _ in range(100):  # 100 validation samples
        val_data.append(target_sentence)
    
    test_data = []
    for _ in range(50):  # 50 test samples
        test_data.append(target_sentence)
    
    # Save to CSV files
    os.makedirs("data", exist_ok=True)
    
    pd.DataFrame({"en": train_data}).to_csv("data/train.csv", index=False)
    pd.DataFrame({"en": val_data}).to_csv("data/val.csv", index=False)
    pd.DataFrame({"en": test_data}).to_csv("data/test.csv", index=False)
    
    print(f"Created MLM training data:")
    print(f"  Target sentence: '{target_sentence}'")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    print(f"  Words in sentence: {len(target_sentence.split())}")

if __name__ == "__main__":
    create_mlm_data()
