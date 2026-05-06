"""
06e_generate_transformer.py
===========================
Generates reconstructed images specifically from the TRANSFORMER model outputs.
Prefix: mlp_v5_transformer
"""

import os
import gc
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
MODELS_DIR = r"D:/PBL_6/DATA_prep/output/models"
SPLITS_DIR = r"D:/PBL_6/DATA_prep/output/splits"
CLIP_DIR   = r"D:/PBL_6/DATA_prep/output/clip_features"
OUTPUT_DIR = r"D:/PBL_6/DATA_prep/output/reconstructions"

# MATCH THIS TO YOUR 05b_train_transformer_v5.py PREFIX
FILE_PREFIX = "mlp_v5_transformer"

# Toggle which test set to run
TEST_SET = "test_scenes_coco" 

SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"
NUM_IMAGES_TO_GEN = 20

# ─────────────────────────────────────────────────────────────
# UTILS (Shared with Rigid for Consistency)
# ─────────────────────────────────────────────────────────────

def build_projection(in_dim, out_dim, device, dtype):
    proj = nn.Linear(in_dim, out_dim).to(device).to(dtype)
    nn.init.eye_(proj.weight[:min(in_dim, out_dim), :min(in_dim, out_dim)])
    return proj

def generate_direct(pipe, proj, pred_embeds, save_dir, device, dtype, n=10):
    os.makedirs(save_dir, exist_ok=True)
    print(f"🎨 Transformer Strategy A: Direct Injection...")
    
    for i in range(min(n, len(pred_embeds))):
        emb = torch.from_numpy(pred_embeds[i]).unsqueeze(0).to(device).to(dtype)
        with torch.no_grad():
            sd_emb = proj(emb).unsqueeze(1).repeat(1, 77, 1)
            image = pipe(prompt_embeds=sd_emb, guidance_scale=7.0).images[0]
            image.save(os.path.join(save_dir, f"recon_direct_{i:03d}.png"))

def generate_retrieve_refine(img2img_pipe, pred_embeds, ref_embeds, split_csv_path, save_dir, device, dtype, n=10):
    os.makedirs(save_dir, exist_ok=True)

    # Use train_mixed because ref_embeds is train_mixed_clip_embeds.npy
    train_csv_path = os.path.join(SPLITS_DIR, "train_mixed.csv")
    df = pd.read_csv(train_csv_path)

    print(f"🔍 Transformer Strategy B: Retrieve & Refine...")
    print(f"Reference embeddings shape: {ref_embeds.shape}")
    print(f"Retrieval CSV rows: {len(df)}")

    if len(ref_embeds) != len(df):
        print("⚠️ WARNING: ref_embeds and train_mixed.csv row count do not match.")
        print("Retrieval indices may be incorrect.")

    ref_norms = ref_embeds / (np.linalg.norm(ref_embeds, axis=1, keepdims=True) + 1e-8)

    for i in range(min(n, len(pred_embeds))):
        pred_norm = pred_embeds[i] / (np.linalg.norm(pred_embeds[i]) + 1e-8)
        similarities = np.dot(ref_norms, pred_norm)
        best_idx = int(np.argmax(similarities))

        row = df.iloc[best_idx]
        img_path = os.path.join(
            r"D:/PBL_6/DATA_prep/Original_Images",
            row["folder"],
            row["filename"]
        )

        print(f"[{i:03d}] best_idx={best_idx}, seed={img_path}")

        if not os.path.exists(img_path):
            print(f"⚠️ Seed image not found: {img_path}")
            continue

        seed_img = Image.open(img_path).convert("RGB").resize((512, 512))

        with torch.no_grad():
            image = img2img_pipe(
                prompt="photorealistic reconstruction of the visual stimulus",
                image=seed_img,
                strength=0.35,
                guidance_scale=5.0,
                num_inference_steps=25,
            ).images[0]

        image.save(os.path.join(save_dir, f"recon_retrieval_{i:03d}.png"))

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    # Transformer predictions use this naming format from script 05b
    pred_path = os.path.join(MODELS_DIR, f"{FILE_PREFIX}_{TEST_SET}_preds.npy")
    ref_path  = os.path.join(CLIP_DIR, "train_mixed_clip_embeds.npy")
    csv_path  = os.path.join(SPLITS_DIR, f"{TEST_SET}.csv")

    if not os.path.exists(pred_path):
        print(f"❌ Prediction file not found: {pred_path}")
        return

    pred_embeds = np.load(pred_path)
    ref_embeds  = np.load(ref_path)
    run_dir = os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}_{TEST_SET}")
    
    # 1. Strategy A
    pipe = StableDiffusionPipeline.from_pretrained(
    SD_MODEL_ID,
    torch_dtype=dtype,
    safety_checker=None,
    requires_safety_checker=False,
).to(device)
    proj = build_projection(pred_embeds.shape[1], 768, device, dtype)
    generate_direct(pipe, proj, pred_embeds, os.path.join(run_dir, "direct"), device, dtype, n=NUM_IMAGES_TO_GEN)
    
    del pipe, proj
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Strategy B
    i2i_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    SD_MODEL_ID,
    torch_dtype=dtype,
    safety_checker=None,
    requires_safety_checker=False,
).to(device)
    generate_retrieve_refine(i2i_pipe, pred_embeds, ref_embeds, csv_path, os.path.join(run_dir, "retrieval"), device, dtype, n=NUM_IMAGES_TO_GEN)

    print(f"\n✅ TRANSFORMER GENERATION COMPLETE: Check {run_dir}")

if __name__ == "__main__":
    main()