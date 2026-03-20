import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel

# ================= CONFIGURATION =================

SPLITS_DIR = r"D:/PBL_6/DATA_prep/output/splits" # Path to your CSV splits (Created in Step 1)
OUTPUT_DIR = r"D:/PBL_6/DATA_prep/output/clip_features" # Path to save the new CLIP features
IMAGES_DIR = r"D:/PBL_6/DATA_prep/Original_Images" # Base path for images (Same as Step 1 config)
MODEL_NAME = "openai/clip-vit-large-patch14"  # Model Selection: We use the one Stable Diffusion v1.5 is based on



def extract_features():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

   
    print(f" Loading CLIP Model ({MODEL_NAME})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   - Running on: {device}")

    model = CLIPVisionModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

   
    csv_files = ["train_mixed.csv", "test_objects_imagenet.csv", "test_scenes_coco.csv"]

    for csv_file in csv_files:
        csv_path = os.path.join(SPLITS_DIR, csv_file)
        if not os.path.exists(csv_path):
            print(f"⚠️  Skipping {csv_file} (File not found)")
            continue

        print(f"\n📂 Processing {csv_file}...")
        df = pd.read_csv(csv_path)

        embeddings = []
        batch_size = 32  # Process 32 images at a time

       
        for i in range(0, len(df), batch_size):  # Loop through images in batches
            batch_df = df.iloc[i : i + batch_size]
            images = []
            valid_indices = []

            for idx, row in batch_df.iterrows():
                # Construct full path based on the 'folder' column from Step 1
                img_path = os.path.join(IMAGES_DIR, row["folder"], row["filename"])

                try:
                    image = Image.open(img_path).convert("RGB")
                    images.append(image)
                    valid_indices.append(idx)
                except Exception as e:
                    print(f"    ❌ Error loading {row['filename']}: {e}")

            if not images:
                continue

            # Pass batch through CLIP
            inputs = processor(images=images, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                batch_embeds = outputs.pooler_output.cpu().numpy()
                embeddings.append(batch_embeds)

            if i % 100 == 0:
                print(f"   - Processed {i}/{len(df)} images...")

        
        if embeddings:
            full_embeddings = np.vstack(embeddings)
            save_name = csv_file.replace(".csv", "_clip_embeds.npy")
            save_path = os.path.join(OUTPUT_DIR, save_name)

            np.save(save_path, full_embeddings)
            print(f"✅ Saved {full_embeddings.shape} to {save_name}")


if __name__ == "__main__":
    extract_features()
