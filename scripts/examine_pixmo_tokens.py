import json
import numpy as np
from collections import Counter

def analyze_tokens(file_path):
    print(f"Analyzing {file_path}")
    
    # Statistics to track
    text_tokens = []  # tokens < 32004
    image_tokens = []  # tokens >= 32004
    sequence_lengths = []
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            tokens = item['tokens']
            sequence_lengths.append(len(tokens))
            
            # Split tokens into text and image
            text_part = [t for t in tokens if t < 32004]
            image_part = [t for t in tokens if t >= 32004]
            
            text_tokens.extend(text_part)
            image_tokens.extend(image_part)
            
            if i == 0:
                print("\nFirst sequence:")
                print(f"Total length: {len(tokens)}")
                print(f"Text tokens: {text_part[:50]}...")
                print(f"Image tokens: {image_part[:50]}...")
                print()
    
    # Analyze text tokens
    text_counter = Counter(text_tokens)
    print("\nText Token Statistics:")
    print(f"Total text tokens: {len(text_tokens)}")
    print(f"Unique text tokens: {len(text_counter)}")
    print(f"Min text token: {min(text_tokens)}")
    print(f"Max text token: {max(text_tokens)}")
    print("\nMost common text tokens:")
    for token, count in text_counter.most_common(10):
        print(f"Token {token}: {count} occurrences")
    
    # Analyze image tokens
    image_counter = Counter(image_tokens)
    print("\nImage Token Statistics:")
    print(f"Total image tokens: {len(image_tokens)}")
    print(f"Unique image tokens: {len(image_counter)}")
    print(f"Min image token: {min(image_tokens)}")
    print(f"Max image token: {max(image_tokens)}")
    print("\nMost common image tokens:")
    for token, count in image_counter.most_common(10):
        print(f"Token {token}: {count} occurrences")
    
    # Sequence length statistics
    print("\nSequence Length Statistics:")
    print(f"Total sequences: {len(sequence_lengths)}")
    print(f"Min length: {min(sequence_lengths)}")
    print(f"Max length: {max(sequence_lengths)}")
    print(f"Mean length: {np.mean(sequence_lengths):.2f}")
    print(f"Median length: {np.median(sequence_lengths):.2f}")
    
    # Token ranges
    all_tokens = text_tokens + image_tokens
    print("\nOverall Token Range:")
    print(f"Min token: {min(all_tokens)}")
    print(f"Max token: {max(all_tokens)}")
    print(f"Total unique tokens: {len(set(all_tokens))}")

if __name__ == "__main__":
    file_path = "/home/ly/d/code/lingua/pixmo_cap_tokens.jsonl"
    analyze_tokens(file_path) 