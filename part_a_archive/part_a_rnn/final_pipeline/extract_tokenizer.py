#!/usr/bin/env python3
"""Extract tokenizer from JSON and save as pickle"""
import json
import pickle
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Load tokenizer from JSON
with open('configs/tokenizer.json', 'r') as f:
    tokenizer_json = f.read()

tokenizer = tokenizer_from_json(tokenizer_json)

# Save as pickle
with open('submission/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("✓ Tokenizer extracted and saved to submission/tokenizer.pkl")
print(f"  Vocabulary size: {len(tokenizer.word_index)}")
