import os
import json
import numpy as np
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer

def process_tokens(output_dir, nchunks=32, seed=42):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the modified tokenizer to get its vocabulary size
    tokenizer = AutoTokenizer.from_pretrained("tokenizers/modified")
    text_vocab_size = len(tokenizer)
    print(f"Loaded tokenizer with vocabulary size: {text_vocab_size}")
    
    # Debug file for token statistics
    debug_file = os.path.join(output_dir, "token_debug.txt")
    
    # Load all data first
    print("Loading data...")
    data = []
    max_token_id = float('-inf')
    min_token_id = float('inf')
    token_counts = {}
    
    input_file = "/home/ly/d/code/lingua/pixmo_cap_tokens.jsonl"
    print(f"Reading from {input_file}")
    
    with open(input_file, "r") as f:
        for line in f:
            item = json.loads(line)
            # Tokens are already properly shifted in the input file
            tokens = item["tokens"]
            
            # Track token statistics
            max_token_id = max(max_token_id, max(tokens))
            min_token_id = min(min_token_id, min(tokens))
            for t in tokens:
                token_counts[t] = token_counts.get(t, 0) + 1
            
            data.append({
                "tokens": tokens
            })
    
    # Write debug information
    with open(debug_file, 'w') as f:
        f.write(f"Text Vocabulary Size: {text_vocab_size}\n")
        f.write(f"Total samples: {len(data)}\n")
        f.write(f"Token Statistics:\n")
        f.write(f"Min token ID: {min_token_id}\n")
        f.write(f"Max token ID: {max_token_id}\n")
        f.write(f"Unique token count: {len(token_counts)}\n")
        f.write("\nToken ID distribution (showing first 20 most common):\n")
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        for token_id, count in sorted_tokens[:20]:
            f.write(f"Token ID {token_id}: {count} occurrences\n")
    
    print(f"Token range: {min_token_id} to {max_token_id}")
    print(f"Debug information written to {debug_file}")
    
    # Shuffle data
    print("Shuffling data...")
    rng = np.random.RandomState(seed)
    indices = np.arange(len(data))
    rng.shuffle(indices)
    
    # Split into chunks
    chunk_size = len(indices) // nchunks
    
    # Write chunks
    print("Writing chunks...")
    for i in tqdm(range(nchunks)):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < nchunks - 1 else len(indices)
        chunk_indices = indices[start_idx:end_idx]
        
        chunk_data = [data[idx] for idx in chunk_indices]
        
        # Save chunk with the expected naming format: *.chunk.*.jsonl
        chunk_path = os.path.join(output_dir, f"pixmo_text_image.chunk.{i:02d}.jsonl")
        with open(chunk_path, 'w') as f:
            for item in chunk_data:
                f.write(json.dumps(item) + '\n')
    
    # Save metadata
    metadata = {
        "num_chunks": nchunks,
        "total_samples": len(data),
        "text_vocab_size": text_vocab_size,
        "min_token_id": int(min_token_id),
        "max_token_id": int(max_token_id)
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Processed {len(data)} samples into {nchunks} chunks")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="Directory to save processed data")
    parser.add_argument("--nchunks", type=int, default=32, help="Number of chunks to split data into")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    process_tokens(args.output_dir, args.nchunks, args.seed) 