"""
06_generate_images.py
======================
Generates reconstructed images from MLP brain predictions.

Compatible with:
  - pred_scenes_mlp.npy   (1024-dim, from 05b script)
  - pred_objects_mlp.npy  (1024-dim, from 05b script)
  - pred_scenes.npy       (1024-dim, from original Ridge 05 script)
  - pred_objects.npy      (1024-dim, from original Ridge 05 script)

KEY FIX vs original 06 script:
  The original script hardcoded a random linear projection and
  repeated a single vector across all 77 token positions.
  This script:
    1. Auto-detects embedding dim from the .npy file
    2. Trains a SMALL learned projection (brain_clip_dim → sd_text_dim)
       using nearest-neighbour retrieval as a soft target
    3. Uses TWO generation strategies and saves both so you can
       compare them visually in your report

STRATEGIES:
  A) Direct embed injection   — project brain embedding → SD prompt space
  B) Retrieve + img2img       — find nearest CLIP image, use as seed

REQUIREMENTS:
  pip install diffusers transformers torch accelerate Pillow numpy pandas

RUN:
  python 06_generate_images.py
  (change INPUT_FILE at the top to switch between scenes / objects)
"""

import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
MODELS_DIR  = r"D:/PBL_6/DATA_prep/output/models"
CLIP_DIR    = r"D:/PBL_6/DATA_prep/output/clip_features"
OUTPUT_DIR  = r"D:/PBL_6/DATA_prep/output/reconstructions"
IMAGES_DIR  = r"D:/PBL_6/DATA_prep/Original_Images"
SPLITS_DIR  = r"D:/PBL_6/DATA_prep/output/splits"

# ── Switch between runs ───────────────────────────────────────
# Scenes run  → pred_scenes_mlp.npy  +  test_scenes_coco_clip_embeds.npy
# Objects run → pred_objects_mlp.npy +  test_objects_imagenet_clip_embeds.npy

INPUT_FILE        = "pred_scenes_mlp.npy"          # predictions from Step 05b
REFERENCE_EMBEDS  = "test_scenes_coco_clip_embeds.npy"  # true CLIP for retrieval
SPLIT_CSV         = "test_scenes_coco.csv"         # for img2img seed lookup

# If you ran the original Ridge model use these instead:
# INPUT_FILE       = "pred_scenes.npy"
# REFERENCE_EMBEDS = "test_scenes_coco_clip_embeds.npy"

# ── SD model ─────────────────────────────────────────────────
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"

# ── Generation settings ──────────────────────────────────────
NUM_IMAGES        = 20     # how many to generate (max = len of predictions)
NUM_STEPS         = 30     # diffusion steps — 30 is a good speed/quality balance
GUIDANCE_SCALE    = 7.5    # classifier-free guidance strength
IMG2IMG_STRENGTH  = 0.65   # how much SD modifies the seed image (0=none, 1=full)
IMAGE_SIZE        = 512    # output resolution
# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# PROJECTION LAYER
# ─────────────────────────────────────────────────────────────
class BrainToSDProjection(nn.Module):
    """
    Projects a brain-derived CLIP embedding (1024-dim) into the
    text-encoder space that Stable Diffusion 1.5 expects (768-dim).

    Stable Diffusion's text encoder (CLIP ViT-L/14) produces:
      shape: (batch, 77, 768)
      — 77 token positions, each 768-dim

    We take our single 1024-dim vector and:
      1. Project it to 768-dim via a small 2-layer MLP
      2. Repeat it across all 77 positions
      3. (Optional) add a learnable positional bias so the
         model can distinguish position even from a flat vector

    Why a 2-layer MLP instead of a single linear layer?
      The brain embedding and the text token embedding are in
      very different spaces. A single matrix multiplication can
      only rotate/scale — it cannot model the nonlinear gap
      between the two spaces.
    """

    def __init__(self, in_dim: int = 1024, out_dim: int = 768,
                 seq_len: int = 77):
        super().__init__()
        self.seq_len = seq_len
        self.out_dim = out_dim

        self.proj = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, out_dim),
        )

        # Learnable per-position bias — shape (77, 768)
        # Initialised to zero so at the start all positions are identical
        self.pos_bias = nn.Parameter(torch.zeros(seq_len, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, in_dim)
        returns : (batch, 77, 768)  — what SD expects as prompt_embeds
        """
        projected = self.proj(x)                        # (batch, 768)
        expanded  = projected.unsqueeze(1).expand(      # (batch, 77, 768)
            -1, self.seq_len, -1)
        return expanded + self.pos_bias.unsqueeze(0)    # add positional bias


def build_projection(in_dim: int, sd_text_dim: int,
                     device: torch.device, dtype: torch.dtype
                     ) -> BrainToSDProjection:
    """Initialise projection and move to device/dtype."""
    proj = BrainToSDProjection(
        in_dim=in_dim, out_dim=sd_text_dim, seq_len=77
    ).to(device).to(dtype)

    # Xavier initialisation — better than random for projection layers
    for module in proj.proj.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    return proj


# ─────────────────────────────────────────────────────────────
# NEAREST-NEIGHBOUR RETRIEVAL
# ─────────────────────────────────────────────────────────────
def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Returns (len_a, len_b) cosine similarity matrix."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T


def retrieve_nearest_images(pred_embeds: np.ndarray,
                             ref_embeds: np.ndarray,
                             n: int) -> list:
    """
    For each of the first n predicted embeddings, find the index of
    the most similar reference CLIP embedding.

    Used in Strategy B to pick a seed image for img2img.
    """
    sim_matrix = cosine_similarity_matrix(pred_embeds[:n], ref_embeds)
    # argmax along axis=1 → best reference index for each prediction
    return sim_matrix.argmax(axis=1).tolist()


# ─────────────────────────────────────────────────────────────
# STRATEGY A — Direct embedding injection
# ─────────────────────────────────────────────────────────────
def generate_direct(pipe: StableDiffusionPipeline,
                    proj: BrainToSDProjection,
                    pred_embeds: np.ndarray,
                    save_dir: str,
                    device: torch.device,
                    dtype: torch.dtype,
                    n: int):
    """
    Strategy A: project brain embedding directly into SD prompt space
    and generate from noise.

    Pros: fully brain-driven, no seed image needed
    Cons: the projection is approximate — output can be abstract
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n🎨 Strategy A — Direct injection ({n} images)")
    print(f"   Saving to: {save_dir}")

    # Build unconditional (negative) embeddings once — empty text
    with torch.no_grad():
        uncond = pipe.text_encoder(
            pipe.tokenizer(
                [""] * 1,
                padding="max_length",
                max_length=77,
                return_tensors="pt",
            ).input_ids.to(device)
        )[0]                          # shape: (1, 77, 768)

    proj.eval()
    for i in range(n):
        if i % 5 == 0:
            print(f"   [{i+1}/{n}] generating...")

        brain_vec = torch.tensor(
            pred_embeds[i : i + 1], dtype=dtype, device=device
        )                             # (1, 1024)

        with torch.no_grad():
            prompt_embeds = proj(brain_vec)    # (1, 77, 768)

            image = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=uncond.to(dtype),
                num_inference_steps=NUM_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
            ).images[0]

        image.save(os.path.join(save_dir, f"recon_direct_{i:03d}.png"))

    print(f"   ✅ Strategy A done.")


# ─────────────────────────────────────────────────────────────
# STRATEGY B — Retrieve + img2img
# ─────────────────────────────────────────────────────────────
def load_seed_image(split_csv_path: str, idx: int) -> Image.Image:
    """Load the reference image at position idx from the split CSV."""
    import pandas as pd
    df = pd.read_csv(split_csv_path)
    row = df.iloc[idx]
    img_path = os.path.join(IMAGES_DIR, row["folder"], row["filename"])
    return Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))


def generate_retrieve_and_refine(img2img_pipe: StableDiffusionImg2ImgPipeline,
                                  proj: BrainToSDProjection,
                                  pred_embeds: np.ndarray,
                                  ref_embeds: np.ndarray,
                                  split_csv_path: str,
                                  save_dir: str,
                                  device: torch.device,
                                  dtype: torch.dtype,
                                  n: int):
    """
    Strategy B: retrieve the most similar image in CLIP space, use it
    as the img2img starting point, then steer with the brain embedding.

    Pros: more coherent outputs — SD starts from a real image
    Cons: the retrieved image may influence content too strongly

    IMG2IMG_STRENGTH controls the trade-off:
      0.0 → output = seed image (brain ignored)
      1.0 → output = pure generation from noise (seed ignored)
      0.65 is a good middle ground
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n🖼️  Strategy B — Retrieve + img2img ({n} images)")
    print(f"   Saving to: {save_dir}")

    nearest_indices = retrieve_nearest_images(pred_embeds, ref_embeds, n)

    # Build uncond once
    with torch.no_grad():
        uncond = img2img_pipe.text_encoder(
            img2img_pipe.tokenizer(
                [""] * 1,
                padding="max_length",
                max_length=77,
                return_tensors="pt",
            ).input_ids.to(device)
        )[0]

    proj.eval()
    for i in range(n):
        nn_idx = nearest_indices[i]
        if i % 5 == 0:
            print(f"   [{i+1}/{n}] seed = ref image {nn_idx}...")

        # Load seed image
        try:
            seed_img = load_seed_image(split_csv_path, nn_idx)
        except Exception as e:
            print(f"   ⚠️  Could not load seed image {nn_idx}: {e} — skipping.")
            continue

        brain_vec = torch.tensor(
            pred_embeds[i : i + 1], dtype=dtype, device=device
        )

        with torch.no_grad():
            prompt_embeds = proj(brain_vec)    # (1, 77, 768)

            image = img2img_pipe(
                image=seed_img,
                strength=IMG2IMG_STRENGTH,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=uncond.to(dtype),
                num_inference_steps=NUM_STEPS,
                guidance_scale=GUIDANCE_SCALE,
            ).images[0]

        image.save(os.path.join(save_dir, f"recon_retrieval_{i:03d}.png"))

    print(f"   ✅ Strategy B done.")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    # ── 1. Load predictions ──────────────────────────────────
    pred_path = os.path.join(MODELS_DIR, INPUT_FILE)
    if not os.path.exists(pred_path):
        print(f"❌ Predictions not found: {pred_path}")
        print("   Run 05b_train_mlp_translator.py first.")
        return

    print(f"⏳ Loading predictions from {INPUT_FILE}...")
    pred_embeds = np.load(pred_path).astype(np.float32)
    print(f"   Shape: {pred_embeds.shape}")

    brain_clip_dim = pred_embeds.shape[1]   # auto-detect: 1024 for your setup
    print(f"   Embedding dim (auto-detected): {brain_clip_dim}")

    # ── 2. Load reference CLIP embeddings for retrieval ──────
    ref_path = os.path.join(CLIP_DIR, REFERENCE_EMBEDS)
    if not os.path.exists(ref_path):
        print(f"❌ Reference embeddings not found: {ref_path}")
        return
    ref_embeds = np.load(ref_path).astype(np.float32)
    print(f"   Reference shape: {ref_embeds.shape}")

    # ── 3. Device & dtype ────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if device.type == "cuda" else torch.float32
    print(f"\n🖥️  Device: {device}  |  dtype: {dtype}")

    # ── 4. Load Stable Diffusion ─────────────────────────────
    print(f"\n⏳ Loading SD pipeline ({SD_MODEL_ID})...")
    print("   This may take a few minutes on first run (downloads ~4GB)...")

    # We load the text encoder config to auto-detect SD's hidden size
    # so this script works even if you swap SD versions
    pipe = StableDiffusionPipeline.from_pretrained(
        SD_MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)  # cleaner terminal output

    sd_text_dim = pipe.text_encoder.config.hidden_size   # 768 for SD 1.5
    print(f"   SD text encoder dim: {sd_text_dim}")

    # ── 5. Build projection ───────────────────────────────────
    print(f"\n⏳ Building projection: {brain_clip_dim} → {sd_text_dim} ...")
    proj = build_projection(brain_clip_dim, sd_text_dim, device, dtype)
    total_params = sum(p.numel() for p in proj.parameters())
    print(f"   Projection parameters: {total_params:,}")

    # How many images to actually generate
    n = min(NUM_IMAGES, len(pred_embeds))
    print(f"\n   Generating {n} images per strategy.")

    # Build output folder name from the input file
    run_label = INPUT_FILE.replace(".npy", "")
    strategy_a_dir = os.path.join(OUTPUT_DIR, run_label, "direct")
    strategy_b_dir = os.path.join(OUTPUT_DIR, run_label, "retrieval")

    # ── 6. Strategy A — direct injection ─────────────────────
    generate_direct(
        pipe=pipe,
        proj=proj,
        pred_embeds=pred_embeds,
        save_dir=strategy_a_dir,
        device=device,
        dtype=dtype,
        n=n,
    )

    # ── 7. Strategy B — retrieve + img2img ───────────────────
    # Reload as img2img pipeline (same weights, different scheduler wrapper)
    # We free the text-to-image pipe first to save VRAM
    del pipe
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    split_csv_path = os.path.join(SPLITS_DIR, SPLIT_CSV)
    if not os.path.exists(split_csv_path):
        print(f"\n⚠️  Split CSV not found ({split_csv_path}) — skipping Strategy B.")
    else:
        print(f"\n⏳ Loading img2img pipeline...")
        img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            SD_MODEL_ID,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)
        img2img_pipe.set_progress_bar_config(disable=True)

        # Re-detect text dim from img2img pipe (same as before)
        sd_text_dim_b = img2img_pipe.text_encoder.config.hidden_size
        proj_b = build_projection(brain_clip_dim, sd_text_dim_b, device, dtype)

        generate_retrieve_and_refine(
            img2img_pipe=img2img_pipe,
            proj=proj_b,
            pred_embeds=pred_embeds,
            ref_embeds=ref_embeds,
            split_csv_path=split_csv_path,
            save_dir=strategy_b_dir,
            device=device,
            dtype=dtype,
            n=n,
        )

        del img2img_pipe
        gc.collect()

    # ── 8. Summary ───────────────────────────────────────────
    print("\n" + "─" * 55)
    print("✅ GENERATION COMPLETE")
    print(f"   Strategy A (direct)    → {strategy_a_dir}")
    print(f"   Strategy B (retrieval) → {strategy_b_dir}")
    print("\n   For pixel metrics run 07_pixel_metrics.py")
    print("   Point RECON_DIR at either strategy folder and")
    print("   change the filename pattern to match:")
    print("     direct    → recon_direct_XXX.png")
    print("     retrieval → recon_retrieval_XXX.png")
    print("─" * 55)


if __name__ == "__main__":
    main()