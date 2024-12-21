import os
import json
from PIL import Image
import torch
from transformers import AutoModel, AutoImageProcessor
import webdataset as wds
from tqdm import tqdm

############################################
# Configuration
############################################
MODEL_HUB = "BAAI/Emu3-VisionTokenizer"
TAR_FILE = "/home/liangyu/code/output.tar"  # Change this to your actual tar file
# TAR_FILE = "/scratch/one_month/current/amirbar/96996.tar"  # Change this to your actual tar file
OUTPUT_DIR = "/home/liangyu/code/tokenized"
BATCH_SIZE = 2

# Split the path and extract the desired components
parts = TAR_FILE.strip("/").split("/")
some_str = parts[-2]
fn = os.path.splitext(parts[-1])[0]  # Remove the .tar extension

# Construct the output path
output_path = os.path.join(OUTPUT_DIR, some_str)
os.makedirs(output_path, exist_ok=True)
output_path = os.path.join(output_path, f"{fn}.jsonl")

############################################
# Model and Processor Initialization
############################################
model = AutoModel.from_pretrained(MODEL_HUB, trust_remote_code=True).eval().cuda()
processor = AutoImageProcessor.from_pretrained(MODEL_HUB, trust_remote_code=True)

vocab_size = model.config.codebook_size
print("Model loaded.")
print(f"Vocabulary Size: {vocab_size}")

############################################
# WebDataset Setup
############################################
dataset = (
    wds.WebDataset(TAR_FILE)
    .decode("pil")  # decode images to PIL
    .to_tuple("jpg;png;jpeg", "json", "__key__")  # extract (image, json, key) pairs
    .map(lambda x: (x[0], x[1], os.path.splitext(x[2])[0].split('/')[-1]))  # Get base filename from key
)

def collate_fn(samples):
    # samples is now a list of (image, metadata_json_str_or_dict, filename) tuples
    images, metas, filenames = zip(*samples)
    return list(images), list(metas), list(filenames)

batched_dataset = dataset.batched(BATCH_SIZE, collation_fn=collate_fn)

############################################
# Processing and Saving
############################################
with open(output_path, 'w') as f_out:
    for images, metas, filenames in tqdm(batched_dataset, desc="Processing batches"):
        # Parse JSON if necessary
        parsed_metas = []
        for m in metas:
            if isinstance(m, str):
                parsed_metas.append(json.loads(m))
            else:
                parsed_metas.append(m)
        metas = parsed_metas

        # Preprocess images
        # The model and processor can handle varying sizes. You can resize if needed.
        # Here we rely on the processor's smart resize by default.
        
        # resize images to 512x512
        images = [image.resize((512, 512)) for image in images]
        
        pixel_values = processor(images, return_tensors="pt")["pixel_values"].cuda()  # shape: (B, C, H, W)

        with torch.no_grad():
            codes = model.encode(pixel_values)  # shape typically (B, h, w)

        # Flatten codes if they are 3D (B, h, w) to (B, h*w)
        if codes.dim() == 3:
            B, H, W = codes.shape
            codes = codes.view(B, H * W)
        elif codes.dim() == 2:
            # Already (B, N)
            pass
        else:
            raise ValueError(f"Unexpected code shape: {codes.shape}")

        codes = codes.cpu().tolist()

        # Verify token range
        for token_list in codes:
            assert all(0 <= t < vocab_size for t in token_list), "Token out of range!"
            assert len(token_list) == 4096, "Token length mismatch!"

        # Write results to jsonl
        for filename, token_list in zip(filenames, codes):
            token_data = {
                "image": filename,  # Use the actual filename instead of metadata
                "tokens": token_list
            }
            f_out.write(json.dumps(token_data) + "\n")

print(f"Tokenization completed! Results saved to {output_path}")