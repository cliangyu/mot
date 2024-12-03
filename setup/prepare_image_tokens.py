import os
import json
import numpy as np
from tqdm import tqdm
import argparse

def prepare_image_tokens(input_file, output_dir, nchunks=32, seed=42):
    """
    Prepare image tokens data by reading from a JSONL file and splitting into chunks.
    Each line in the input file should contain {"image": "image_name.jpg", "tokens": [...]}
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read all tokens from input file
    print(f"Reading tokens from {input_file}")
    all_data = []
    with open(input_file, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
    
    # Shuffle data
    rng = np.random.RandomState(seed)
    rng.shuffle(all_data)
    
    # Split into chunks
    chunk_size = len(all_data) // nchunks
    
    # Write chunks
    for i in tqdm(range(nchunks), desc="Writing chunks"):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < nchunks - 1 else len(all_data)
        chunk_data = all_data[start_idx:end_idx]
        
        # Save chunk with the expected naming format: *.chunk.*.jsonl
        chunk_path = os.path.join(output_dir, f"image_tokens.chunk.{i:02d}.jsonl")
        with open(chunk_path, 'w') as f:
            for item in chunk_data:
                f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Input JSONL file containing image tokens")
    parser.add_argument("output_dir", help="Directory to save processed data")
    parser.add_argument("--nchunks", type=int, default=32, help="Number of chunks to split data into")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    prepare_image_tokens(args.input_file, args.output_dir, args.nchunks, args.seed) 