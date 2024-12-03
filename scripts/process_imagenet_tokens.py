import os
import json
import shutil
from tqdm import tqdm

# Create data directory structure
data_dir = "data"
imagenet_dir = os.path.join(data_dir, "image_tokens")
os.makedirs(imagenet_dir, exist_ok=True)

# Process tokens in chunks to avoid memory issues
CHUNK_SIZE = 1000
current_chunk = []
chunk_id = 0

print("Processing imagenet validation tokens...")
with open("imagenet_val_tokens.jsonl", "r") as f:
    for line in tqdm(f):
        data = json.loads(line)
        current_chunk.append({
            "tokens": data["tokens"],
            "image": data["image"]
        })
        
        if len(current_chunk) >= CHUNK_SIZE:
            # Save chunk
            chunk_file = os.path.join(imagenet_dir, f"chunk_{chunk_id:04d}.jsonl")
            with open(chunk_file, "w") as cf:
                for item in current_chunk:
                    cf.write(json.dumps(item) + "\n")
            current_chunk = []
            chunk_id += 1

# Save any remaining items
if current_chunk:
    chunk_file = os.path.join(imagenet_dir, f"chunk_{chunk_id:04d}.jsonl")
    with open(chunk_file, "w") as cf:
        for item in current_chunk:
            cf.write(json.dumps(item) + "\n")

print(f"Processed {chunk_id + 1} chunks of data")

# Create a metadata file
metadata = {
    "num_chunks": chunk_id + 1,
    "chunk_size": CHUNK_SIZE,
    "total_images": (chunk_id * CHUNK_SIZE) + len(current_chunk)
} 