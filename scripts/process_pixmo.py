import os
import os.path as osp
import json
import random
from PIL import Image
import torch
import requests
from datasets import load_dataset
from transformers import AutoModel, AutoImageProcessor, LlamaTokenizer
from tqdm import tqdm

# Constants
MODEL_HUB = "BAAI/Emu3-VisionTokenizer"
LLAMA_MODEL = "meta-llama/Llama-2-7b-hf"  # We'll use this tokenizer as base
OUTPUT_FILE = "pixmo_cap_tokens.jsonl"
SPECIAL_TOKENS = ["<boi>", "<eoi>", "<bot>", "<eot>"]
IMAGE_TOKEN_SIZE = 4096
UNDERSTANDING_RATIO = 0.2  # 20% for understanding tasks

class PixmoProcessor:
    def __init__(self):
        # Initialize vision tokenizer
        self.vision_model = AutoModel.from_pretrained(MODEL_HUB, trust_remote_code=True).eval().cuda()
        self.vision_processor = AutoImageProcessor.from_pretrained(MODEL_HUB, trust_remote_code=True)
        
        # Get image vocabulary size from model config
        self.image_vocab_size = self.vision_model.config.codebook_size
        print(f"Image vocabulary size: {self.image_vocab_size}")
        
        # Initialize text tokenizer (modified Llama tokenizer)
        self.text_tokenizer = LlamaTokenizer.from_pretrained(LLAMA_MODEL)
        original_vocab_size = len(self.text_tokenizer)
        print(f"Original text vocabulary size: {original_vocab_size}")
        
        # Add special tokens and get their IDs
        special_tokens_dict = {'additional_special_tokens': SPECIAL_TOKENS}
        num_added = self.text_tokenizer.add_special_tokens(special_tokens_dict)
        print(f"Added {num_added} special tokens")
        
        # Get the final text vocabulary size after adding special tokens
        self.text_vocab_size = len(self.text_tokenizer)
        print(f"Final text vocabulary size: {self.text_vocab_size}")
        
        # Verify and store special token IDs
        self.special_token_ids = {}
        for token in SPECIAL_TOKENS:
            # Get token ID and verify it's a single token
            token_ids = self.text_tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(f"Special token {token} was encoded into {len(token_ids)} tokens: {token_ids}")
            self.special_token_ids[token] = token_ids[0]
            print(f"Special token {token} -> ID {token_ids[0]}")
        
        # Set image token shift based on text vocabulary size
        self.image_token_shift = self.text_vocab_size
        print(f"Image tokens will be shifted by {self.image_token_shift}")
        
        # Save the modified tokenizer
        os.makedirs("tokenizers/modified", exist_ok=True)
        self.text_tokenizer.save_pretrained("tokenizers/modified")
        
        # Create image storage directory
        self.image_dir = "data/pixmo_images"
        os.makedirs(self.image_dir, exist_ok=True)
        
        # Load dataset
        self.dataset = load_dataset("allenai/pixmo-cap")
        
    def download_image(self, url, image_id):
        try:
            # Create a deterministic filename based on image_id
            image_path = osp.join(self.image_dir, f"image_{image_id}.jpg")
            
            # If image already exists, skip download
            if osp.exists(image_path):
                return image_path
                
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            with open(image_path, 'wb') as f:
                f.write(response.content)
            return image_path
            
        except Exception as e:
            print(f"Error downloading image from {url}: {str(e)}")
            return None

    def process_image(self, image_path):
        try:
            # Load and resize image to 512x512
            image = Image.open(image_path)
            image = image.resize((512, 512), Image.Resampling.LANCZOS)
            processed_image = self.vision_processor(image, return_tensors="pt")["pixel_values"].cuda()
            
            with torch.no_grad():
                codes = self.vision_model.encode(processed_image)
            
            # Convert tokens to list and shift by text vocab size
            tokens_flat = codes.squeeze().reshape(-1).cpu().tolist()
            tokens_flat = [int(t) + self.image_token_shift for t in tokens_flat]
            
            # Verify token range
            assert all(self.image_token_shift <= t < self.image_token_shift + self.image_vocab_size for t in tokens_flat), \
                   f"Token values outside expected range for {image_path}!"
            
            return tokens_flat[:IMAGE_TOKEN_SIZE]  # Ensure exactly 4096 tokens
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None

    def process_text(self, text):
        tokens = self.text_tokenizer.encode(text, add_special_tokens=False)
        # Verify text tokens are within text vocabulary range
        assert all(t < self.text_vocab_size for t in tokens), f"Text token values outside vocabulary range!"
        return tokens

    def process_dataset(self):
        with open(OUTPUT_FILE, 'w') as f:
            for idx, item in enumerate(tqdm(self.dataset['train'], desc="Processing dataset")):
                try:
                    # Download and process image
                    image_path = self.download_image(item['image_url'], idx)
                    if not image_path:
                        continue
                        
                    image_tokens = self.process_image(image_path)
                    if not image_tokens:
                        continue
                        
                    # Process caption
                    text_tokens = self.process_text(item['caption'])
                    
                    # Randomly decide task type (understanding vs generation)
                    is_understanding = random.random() < UNDERSTANDING_RATIO
                    
                    # Use verified special token IDs
                    if is_understanding:
                        # Image first (understanding task)
                        combined_tokens = (
                            [self.special_token_ids['<boi>']] +
                            image_tokens +
                            [self.special_token_ids['<eoi>'], self.special_token_ids['<bot>']] +
                            text_tokens +
                            [self.special_token_ids['<eot>']]
                        )
                    else:
                        # Text first (generation task)
                        combined_tokens = (
                            [self.special_token_ids['<bot>']] +
                            text_tokens +
                            [self.special_token_ids['<eot>'], self.special_token_ids['<boi>']] +
                            image_tokens +
                            [self.special_token_ids['<eoi>']]
                        )
                    
                    # Save to JSONL
                    token_data = {
                        "tokens": combined_tokens,
                        "task_type": "understanding" if is_understanding else "generation",
                        "original_caption": item['caption'],
                        "image_url": item['image_url'],
                        "image_path": image_path
                    }
                    f.write(json.dumps(token_data) + '\n')
                    
                except Exception as e:
                    print(f"Error processing item {idx}: {str(e)}")
                    continue

if __name__ == "__main__":
    processor = PixmoProcessor()
    processor.process_dataset() 