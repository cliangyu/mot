import os
import json
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import argparse

def prepare_data(output_dir, dataset_name, nchunks=32, seed=42):
    """
    Prepare either Pixmo or BLIP3-KALE dataset
    """
    # Create dataset subdirectory
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Load dataset
    if dataset_name == "pixmo":
        dataset = load_dataset("allenai/pixmo-cap")
        captions = dataset["train"]["caption"]
    elif dataset_name == "blip3":
        dataset = load_dataset("Salesforce/blip3-kale")
        captions = dataset["train"]["caption"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Shuffle data
    rng = np.random.RandomState(seed)
    indices = np.arange(len(captions))
    rng.shuffle(indices)
    
    # Split into chunks
    chunk_size = len(indices) // nchunks
    
    # Write chunks
    for i in tqdm(range(nchunks), desc=f"Processing {dataset_name}"):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < nchunks - 1 else len(indices)
        chunk_indices = indices[start_idx:end_idx]
        
        chunk_data = [{"text": captions[idx]} for idx in chunk_indices]
        
        # Save chunk with the expected naming format: *.chunk.*.jsonl
        chunk_path = os.path.join(dataset_dir, f"{dataset_name}.chunk.{i:02d}.jsonl")
        with open(chunk_path, 'w') as f:
            for item in chunk_data:
                f.write(json.dumps(item) + '\n')

def prepare_all_datasets(output_dir, nchunks=32, seed=42):
    """Prepare all datasets"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare each dataset
    prepare_data(output_dir, "pixmo", nchunks, seed)
    prepare_data(output_dir, "blip3", nchunks, seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="Directory to save processed data")
    parser.add_argument("--nchunks", type=int, default=32, help="Number of chunks to split data into")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    prepare_all_datasets(args.output_dir, args.nchunks, args.seed) 