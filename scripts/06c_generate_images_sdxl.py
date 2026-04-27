"""
06_generate_images_sdxl.py
===========================
Upgraded generation script using Stable Diffusion XL (SDXL)
instead of SD 1.5.

WHY SDXL IS BETTER FOR YOUR PROJECT:
  - Generates 1024x1024 images (vs 512x512 in SD 1.5)
  - Much better detail, texture, and scene coherence
  - Uses TWO text encoders:
      CLIP-L  → 768-dim
      OpenCLIP-G → 1280-dim
    These are concatenated → 2048-dim total conditioning
  - Your brain predictions are 1024-dim, so we project to 2048-dim
  - Same diffusers API as SD 1.5 — just different model + conditioning

WHAT CHANGES VS SD 1.5 SCRIPT:
  1. Model ID → stabilityai/stable-diffusion-xl-base-1.0
  2. Projection layer: 1024 → 2048 (not 1024 → 768)
  3. prompt_embeds shape: (1, 77, 2048) for SDXL
  4. Also need pooled_prompt_embeds: (1, 1280) — explained below
  5. Output size: 1024x1024

POOLED PROMPT EMBEDS EXPLAINED (simple version):
  SDXL uses two conditioning signals:
    - Full sequence embeds (1, 77, 2048) — per-token context
    - Pooled embeds     (1, 1280)        — single global summary
  SD 1.5 only uses the first one.
  We generate both from our brain prediction.

REQUIREMENTS:
  pip install diffusers transformers torch accelerate Pillow numpy pandas

RUN:
  python 06_generate_images_sdxl.py
"""

import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
MODELS_DIR = r"D:/PBL_6/DATA_prep/output/models"
CLIP_DIR   = r"D:/PBL_6/DATA_prep/output/clip_features"
OUTPUT_DIR = r"D:/PBL_6/DATA_prep/output/reconstructions"
IMAGES_DIR = r"D:/PBL_6/DATA_prep/Original_Images"
SPLITS_DIR = r"D:/PBL_6/DATA_prep/output/splits"

# ── Switch between runs ───────────────────────────────────────
INPUT_FILE       = "pred_scenes_mlp.npy"
REFERENCE_EMBEDS = "test_scenes_coco_clip_embeds.npy"
SPLIT_CSV        = "test_scenes_coco.csv"

# For objects:
# INPUT_FILE       = "pred_objects_mlp.npy"
# REFERENCE_EMBEDS = "test_objects_imagenet_clip_embeds.npy"
# SPLIT_CSV        = "test_objects_imagenet.csv"

# ── SDXL model ────────────────────────────────────────────────
SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# ── Generation settings ──────────────────────────────────────
NUM_IMAGES       = 20
NUM_STEPS        = 30     # SDXL needs fewer steps than SD 1.5
GUIDANCE_SCALE   = 7.5
IMG2IMG_STRENGTH = 0.65
IMAGE_SIZE       = 1024   # SDXL native resolution

# ── SDXL conditioning dims (do not change) ───────────────────
# SDXL text encoder 1 (CLIP-L)     → hidden_size = 768
# SDXL text encoder 2 (OpenCLIP-G) → hidden_size = 1280
# Combined (concatenated along dim=-1) → 2048
SDXL_SEQUENCE_DIM = 2048   # shape of prompt_embeds last dim
SDXL_POOLED_DIM   = 1280   # shape of pooled_prompt_embeds last dim
SEQ_LEN           = 77     # token sequence length
# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# PROJECTION LAYER FOR SDXL
# ─────────────────────────────────────────────────────────────
class BrainToSDXLProjection(nn.Module):
    """
    Projects a brain-derived CLIP embedding (1024-dim) into
    SDXL's dual-encoder conditioning space.

    SDXL needs TWO things:
      1. prompt_embeds:        shape (batch, 77, 2048)
         → per-token sequence context (both encoders concatenated)

      2. pooled_prompt_embeds: shape (batch, 1280)
         → a single global summary vector (from encoder 2 only)

    We create both from the same 1024-dim brain prediction
    using two separate projection heads.

    Why two heads?
      They serve different roles in the UNet attention mechanism.
      The sequence embeds control fine-grained spatial details.
      The pooled embeds control the global style/content.
    """

    def __init__(self, in_dim: int = 1024,
                 seq_dim: int = 2048,
                 pooled_dim: int = 1280,
                 seq_len: int = 77):
        super().__init__()
        self.seq_len = seq_len

        
        self.seq_head = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, seq_dim),
        )

        # Head 2: brain → pooled embedding (global summary)
        # 1024 → 512 → 1280
        self.pooled_head = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, pooled_dim),
        )

        # Learnable positional bias across 77 token positions
        self.pos_bias = nn.Parameter(torch.zeros(seq_len, seq_dim))

        self._init_weights()

    def _init_weights(self):
        """Xavier init for stable training start."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor):
        """
        x : (batch, 1024)

        Returns:
          prompt_embeds        : (batch, 77, 2048)
          pooled_prompt_embeds : (batch, 1280)
        """
        # Sequence path
        seq = self.seq_head(x)                        # (batch, 2048)
        seq = seq.unsqueeze(1).expand(-1, self.seq_len, -1)  # (batch, 77, 2048)
        seq = seq + self.pos_bias.unsqueeze(0)        # add positional bias

        # Pooled path
        pooled = self.pooled_head(x)                  # (batch, 1280)

        return seq, pooled


# ─────────────────────────────────────────────────────────────
# NEAREST NEIGHBOUR RETRIEVAL (same as SD 1.5 script)
# ─────────────────────────────────────────────────────────────
def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_n @ b_n.T


def retrieve_nearest(pred: np.ndarray, ref: np.ndarray, n: int) -> list:
    return cosine_sim(pred[:n], ref).argmax(axis=1).tolist()


def load_seed_image(split_csv: str, idx: int) -> Image.Image:
    import pandas as pd
    df  = pd.read_csv(split_csv)
    row = df.iloc[idx]
    path = os.path.join(IMAGES_DIR, row["folder"], row["filename"])
    return Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))


# ─────────────────────────────────────────────────────────────
# BUILD UNCONDITIONED EMBEDDINGS FOR SDXL
# ─────────────────────────────────────────────────────────────
def get_sdxl_uncond(pipe, device, dtype):
    """
    SDXL needs unconditioned (negative) embeddings too.
    We encode an empty string through both text encoders
    to get the null conditioning vectors.

    Returns:
      uncond_seq    : (1, 77, 2048)
      uncond_pooled : (1, 1280)
    """
    tokenizer_1 = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    encoder_1   = pipe.text_encoder
    encoder_2   = pipe.text_encoder_2

    with torch.no_grad():
        # Encode empty string through encoder 1 (CLIP-L → 768)
        tokens_1 = tokenizer_1(
            [""], padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        ).input_ids.to(device)
        out_1 = encoder_1(tokens_1, output_hidden_states=True)
        # SDXL uses the penultimate hidden state, not the final output
        enc1_seq = out_1.hidden_states[-2].to(dtype)  # (1, 77, 768)

        # Encode empty string through encoder 2 (OpenCLIP-G → 1280)
        tokens_2 = tokenizer_2(
            [""], padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        ).input_ids.to(device)
        out_2 = encoder_2(tokens_2, output_hidden_states=True)
        enc2_seq    = out_2.hidden_states[-2].to(dtype)  # (1, 77, 1280)
        enc2_pooled = out_2[0].to(dtype)                 # (1, 1280)

    # Concatenate along embedding dim: (1, 77, 768+1280) = (1, 77, 2048)
    uncond_seq    = torch.cat([enc1_seq, enc2_seq], dim=-1)
    uncond_pooled = enc2_pooled

    return uncond_seq, uncond_pooled


# ─────────────────────────────────────────────────────────────
# STRATEGY A — DIRECT INJECTION (SDXL)
# ─────────────────────────────────────────────────────────────
def generate_direct_sdxl(pipe, proj, pred_embeds,
                          save_dir, device, dtype, n):
    """
    Strategy A for SDXL: project brain embedding into dual
    conditioning format and generate from noise.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n🎨 Strategy A — Direct injection SDXL ({n} images)")

    uncond_seq, uncond_pooled = get_sdxl_uncond(pipe, device, dtype)
    proj.eval()

    for i in range(n):
        if i % 5 == 0:
            print(f"   [{i+1}/{n}] generating...")

        brain_vec = torch.tensor(
            pred_embeds[i:i+1], dtype=dtype, device=device
        )  # (1, 1024)

        with torch.no_grad():
            prompt_seq, prompt_pooled = proj(brain_vec)
            # prompt_seq    : (1, 77, 2048)
            # prompt_pooled : (1, 1280)

            image = pipe(
                prompt_embeds=prompt_seq,
                pooled_prompt_embeds=prompt_pooled,
                negative_prompt_embeds=uncond_seq,
                negative_pooled_prompt_embeds=uncond_pooled,
                num_inference_steps=NUM_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
            ).images[0]

        image.save(os.path.join(save_dir, f"recon_direct_{i:03d}.png"))

    print("   ✅ Strategy A done.")


# ─────────────────────────────────────────────────────────────
# STRATEGY B — RETRIEVE + IMG2IMG (SDXL)
# ─────────────────────────────────────────────────────────────
def generate_retrieve_sdxl(img2img_pipe, proj, pred_embeds,
                            ref_embeds, split_csv, save_dir,
                            device, dtype, n):
    """
    Strategy B for SDXL: nearest-neighbour retrieval as seed,
    then img2img steered by brain embedding.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n🖼️  Strategy B — Retrieve + img2img SDXL ({n} images)")

    nearest = retrieve_nearest(pred_embeds, ref_embeds, n)
    uncond_seq, uncond_pooled = get_sdxl_uncond(img2img_pipe, device, dtype)
    proj.eval()

    for i in range(n):
        nn_idx = nearest[i]
        if i % 5 == 0:
            print(f"   [{i+1}/{n}] seed = ref image {nn_idx}...")

        try:
            seed_img = load_seed_image(split_csv, nn_idx)
        except Exception as e:
            print(f"   ⚠️  Seed load failed for idx {nn_idx}: {e}")
            continue

        brain_vec = torch.tensor(
            pred_embeds[i:i+1], dtype=dtype, device=device
        )

        with torch.no_grad():
            prompt_seq, prompt_pooled = proj(brain_vec)

            image = img2img_pipe(
                image=seed_img,
                strength=IMG2IMG_STRENGTH,
                prompt_embeds=prompt_seq,
                pooled_prompt_embeds=prompt_pooled,
                negative_prompt_embeds=uncond_seq,
                negative_pooled_prompt_embeds=uncond_pooled,
                num_inference_steps=NUM_STEPS,
                guidance_scale=GUIDANCE_SCALE,
            ).images[0]

        image.save(os.path.join(save_dir, f"recon_retrieval_{i:03d}.png"))

    print("   ✅ Strategy B done.")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():

    # ── 1. Load predictions ──────────────────────────────────
    pred_path = os.path.join(MODELS_DIR, INPUT_FILE)
    if not os.path.exists(pred_path):
        print(f"❌ Not found: {pred_path}")
        print("   Run 05b_train_mlp_translator.py first.")
        return

    print(f"⏳ Loading predictions from {INPUT_FILE}...")
    pred_embeds    = np.load(pred_path).astype(np.float32)
    brain_clip_dim = pred_embeds.shape[1]
    print(f"   Shape: {pred_embeds.shape}")
    print(f"   Brain embedding dim: {brain_clip_dim}")

    # ── 2. Load reference CLIP embeds for retrieval ──────────
    ref_path = os.path.join(CLIP_DIR, REFERENCE_EMBEDS)
    if not os.path.exists(ref_path):
        print(f"❌ Not found: {ref_path}")
        return
    ref_embeds = np.load(ref_path).astype(np.float32)
    print(f"   Reference shape: {ref_embeds.shape}")

    # ── 3. Device & dtype ────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if device.type == "cuda" else torch.float32
    print(f"\n🖥️  Device: {device}  |  dtype: {dtype}")

    if device.type == "cpu":
        print("   ⚠️  WARNING: SDXL on CPU will be extremely slow.")
        print("   ⚠️  Recommended: NVIDIA GPU with 8GB+ VRAM.")

    # ── 4. Build projection layer ────────────────────────────
    print(f"\n⏳ Building projection: {brain_clip_dim} → "
          f"seq({SDXL_SEQUENCE_DIM}) + pooled({SDXL_POOLED_DIM})")
    proj = BrainToSDXLProjection(
        in_dim=brain_clip_dim,
        seq_dim=SDXL_SEQUENCE_DIM,
        pooled_dim=SDXL_POOLED_DIM,
        seq_len=SEQ_LEN,
    ).to(device).to(dtype)

    total_params = sum(p.numel() for p in proj.parameters())
    print(f"   Projection parameters: {total_params:,}")

    n          = min(NUM_IMAGES, len(pred_embeds))
    run_label  = INPUT_FILE.replace(".npy", "") + "_sdxl"
    dir_a      = os.path.join(OUTPUT_DIR, run_label, "direct")
    dir_b      = os.path.join(OUTPUT_DIR, run_label, "retrieval")
    split_csv  = os.path.join(SPLITS_DIR, SPLIT_CSV)

    # ── 5. Load SDXL text-to-image pipeline ──────────────────
    print(f"\n⏳ Loading SDXL ({SDXL_MODEL_ID})...")
    print("   First run downloads ~6.5GB — this is normal.")
    print("   Subsequent runs load from cache in seconds.")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if dtype == torch.float16 else None,
    )
    #.to(device)
    pipe.set_progress_bar_config(disable=True)

    # Memory optimisation — important for 8GB GPUs
    # Offloads parts of the model to CPU when not needed
    if device.type == "cuda":
        pipe.enable_model_cpu_offload()
        # pipe.enable_xformers_memory_efficient_attention()  # uncomment if xformers installed

    # ── 6. Strategy A ────────────────────────────────────────
    generate_direct_sdxl(
        pipe=pipe, proj=proj, pred_embeds=pred_embeds,
        save_dir=dir_a, device=device, dtype=dtype, n=n,
    )

    # ── 7. Free VRAM, load img2img pipeline ──────────────────
    del pipe
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if not os.path.exists(split_csv):
        print(f"\n⚠️  Split CSV not found — skipping Strategy B.")
    else:
        print(f"\n⏳ Loading SDXL img2img pipeline...")
        img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            SDXL_MODEL_ID,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if dtype == torch.float16 else None,
        ).to(device)
        img2img_pipe.set_progress_bar_config(disable=True)

        if device.type == "cuda":
            img2img_pipe.enable_model_cpu_offload()

        # Build a fresh projection for img2img pipe
        proj_b = BrainToSDXLProjection(
            in_dim=brain_clip_dim,
            seq_dim=SDXL_SEQUENCE_DIM,
            pooled_dim=SDXL_POOLED_DIM,
            seq_len=SEQ_LEN,
        ).to(device).to(dtype)

        generate_retrieve_sdxl(
            img2img_pipe=img2img_pipe, proj=proj_b,
            pred_embeds=pred_embeds, ref_embeds=ref_embeds,
            split_csv=split_csv, save_dir=dir_b,
            device=device, dtype=dtype, n=n,
        ).to(device).to(dtype)

        del img2img_pipe
        gc.collect()

    # ── 8. Summary ───────────────────────────────────────────
    print("\n" + "─" * 55)
    print("✅ SDXL GENERATION COMPLETE")
    print(f"   Images saved at 1024x1024 resolution")
    print(f"   Strategy A (direct)    → {dir_a}")
    print(f"   Strategy B (retrieval) → {dir_b}")
    print("\n   For pixel metrics update 07_pixel_metrics.py:")
    print("   Change IMAGE_SIZE from 512 to 1024 for SDXL outputs")
    print("─" * 55)


if __name__ == "__main__":
    main()