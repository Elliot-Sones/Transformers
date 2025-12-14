"""
Script to add single words as dedicated tokens to the tokenizer.
This prevents common words from being split into subwords.
"""
import json
import csv
import os

# Paths
TOKENIZER_PATH = "tokenizer.json"
OUTPUT_PATH = "tokenizer_with_words.json"
SINGLE_WORDS_PATH = "data/single_words_expanded.csv"

# Load the tokenizer
with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
    tokenizer = json.load(f)

# Get current vocab
vocab = tokenizer['model']['vocab']
current_max_id = max(vocab.values())
print(f"Current vocab size: {len(vocab)}")
print(f"Max token ID: {current_max_id}")

# Read single words
words_to_add = set()
with open(SINGLE_WORDS_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        en_word = row['en'].strip().lower()
        fr_word = row['fr'].strip().lower()
        # Only add single words (no spaces)
        if ' ' not in en_word:
            words_to_add.add(en_word)
        if ' ' not in fr_word:
            words_to_add.add(fr_word)

# Filter out words already in vocab
new_words = []
for word in sorted(words_to_add):
    if word not in vocab:
        new_words.append(word)

print(f"\nWords to add: {len(new_words)}")
print(f"Sample new words: {new_words[:20]}")

# Add new words to vocab
for word in new_words:
    current_max_id += 1
    vocab[word] = current_max_id

print(f"\nNew vocab size: {len(vocab)}")

# Save updated tokenizer
tokenizer['model']['vocab'] = vocab
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(tokenizer, f, ensure_ascii=False, indent=2)

print(f"Saved updated tokenizer to: {OUTPUT_PATH}")

# Test the new tokenizer
from tokenizers import Tokenizer
tok = Tokenizer.from_file(OUTPUT_PATH)

test_words = ['hello', 'hi', 'hey', 'goodbye', 'bonjour', 'salut', 'merci']
print("\nTokenization test:")
for word in test_words:
    encoded = tok.encode(word)
    print(f"  {word} -> {encoded.tokens}")
