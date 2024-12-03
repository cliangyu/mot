import os
import os.path as osp
import json
from PIL import Image
import torch
from transformers import AutoModel, AutoImageProcessor
from tqdm import tqdm

MODEL_HUB = "BAAI/Emu3-VisionTokenizer"

model = AutoModel.from_pretrained(MODEL_HUB, trust_remote_code=True).eval().cuda()
processor = AutoImageProcessor.from_pretrained(MODEL_HUB, trust_remote_code=True)

# Print full config to inspect available attributes
print("Model config:", model.config)
print("-" * 50 + "\n")

# Print vocabulary size
vocab_size = model.config.codebook_size
print(f"Visual Tokenizer Vocabulary Size: {vocab_size}")
print("-" * 50 + "\n")

# Path to your images
IMAGE_DIR = "/home/ly/d/code/DiGIT/ILSVRC2012_img_val"

# Get list of all image files
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()  # Ensure consistent ordering

print(f"Found {len(image_files)} images to process")

# Process all images
with open('imagenet_val_tokens.jsonl', 'w') as f:
    for img_file in tqdm(image_files, desc="Processing images"):
        try:
            # Load and resize image to 512x512
            image = Image.open(osp.join(IMAGE_DIR, img_file))
            image = image.resize((512, 512), Image.Resampling.LANCZOS)
            processed_image = processor(image, return_tensors="pt")["pixel_values"].cuda()
            
            with torch.no_grad():
                # encode
                codes = model.encode(processed_image)
            
            # Convert tokens to list and verify range
            tokens_flat = codes.squeeze().reshape(-1).cpu().tolist()
            tokens_flat = [int(t) for t in tokens_flat]
            assert all(0 <= t < vocab_size for t in tokens_flat), f"Token values outside vocabulary range for {img_file}!"
            
            # Save to JSONL
            token_data = {
                "image": img_file,
                "tokens": tokens_flat
            }
            f.write(json.dumps(token_data) + '\n')
            
        except Exception as e:
            print(f"\nError processing {img_file}: {str(e)}")
            continue

print("\nTokenization completed!")
print("Tokens have been saved to 'imagenet_val_tokens.jsonl'") 