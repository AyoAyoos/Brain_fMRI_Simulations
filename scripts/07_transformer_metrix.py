import os
import math
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = r"D:/PBL_6/DATA_prep"

# Ground-truth images base path (Original BOLD5000 Images)
IMAGES_DIR = os.path.join(BASE_DIR, "Original_Images")

# CSV split used for the test set
SPLITS_DIR = os.path.join(BASE_DIR, "output", "splits")
SPLIT_CSV = "test_scenes_coco.csv"  # Switch to "test_objects_imagenet.csv" for ImageNet

# Reconstruction output folder from 06e (where the PNGs are stored)
RECON_BASE_DIR = os.path.join(BASE_DIR, "output", "reconstructions")

# Prefix/Label matching your Phase 5 Transformer run (FILE_PREFIX + TEST_SET)
RUN_LABEL = "mlp_v5_transformer_test_scenes_coco"

# Output metrics file path
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "metrics")
OUTPUT_METRICS_CSV = os.path.join(OUTPUT_DIR, "transformer_pixel_metrics(2).csv")

# Standard Stable Diffusion output resolution
IMAGE_SIZE = 512

# Strategies generated in script 06e
STRATEGIES = {
    "direct": {
        "folder": "direct",
        "filename_pattern": "recon_direct_{:03d}.png",
    },
    "retrieval": {
        "folder": "retrieval",
        "filename_pattern": "recon_retrieval_{:03d}.png",
    }
}

# ============================================================
# METRIC UTILITIES
# ============================================================

def calculate_psnr(img1, img2):
    """Calculates Peak Signal-to-Noise Ratio (PSNR)"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def evaluate_strategy(df, strategy_name, config):
    """Iterates through images of a specific strategy and calculates metrics"""
    print(f"\n📊 Evaluating Transformer Strategy: {strategy_name.upper()}")
    
    recon_dir = os.path.join(RECON_BASE_DIR, RUN_LABEL, config["folder"])
    
    if not os.path.exists(recon_dir):
        print(f"   ⚠️ Folder not found: {recon_dir}")
        return []

    results = []
    
    # We evaluate based on the index (000, 001...) used during generation
    for i, row in df.iterrows():
        # Stop if the image was never generated (e.g., if you only generated top 20)
        recon_path = os.path.join(recon_dir, config["filename_pattern"].format(i))
        
        if not os.path.exists(recon_path):
            continue

        # Load Ground Truth path from the split CSV
        gt_path = os.path.join(IMAGES_DIR, row["folder"], row["filename"])
        
        try:
            # Load and ensure identical sizes for pixel comparison
            gt_img = Image.open(gt_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            rc_img = Image.open(recon_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))

            # Convert to numpy arrays
            gt_arr = np.array(gt_img)
            rc_arr = np.array(rc_img)

            # 1. MSE (Pixel-wise mean squared error)
            cur_mse = mean_squared_error(gt_arr.ravel(), rc_arr.ravel())
            
            # 2. PSNR
            cur_psnr = calculate_psnr(gt_arr, rc_arr)

            # 3. SSIM (Structural Similarity - calculated on luminance)
            gt_gray = np.array(gt_img.convert("L"))
            rc_gray = np.array(rc_img.convert("L"))
            cur_ssim = ssim(gt_gray, rc_gray)

            results.append({
                "image_name": row["filename"],
                "strategy": strategy_name,
                "mse": round(cur_mse, 2),
                "psnr": round(cur_psnr, 2),
                "ssim": round(cur_ssim, 4)
            })

        except Exception as e:
            print(f"   ❌ Error processing image {i} ({row['filename']}): {e}")

    print(f"   ✅ Successfully processed {len(results)} images.")
    return results

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("🚀 Running Pixel-Based Evaluation for Brain-Transformer...")

    csv_path = os.path.join(SPLITS_DIR, SPLIT_CSV)
    if not os.path.exists(csv_path):
        print(f"❌ Error: Split CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    all_results = []

    # Evaluate both Direct and Retrieval strategies
    for strategy_name, strategy_config in STRATEGIES.items():
        strategy_results = evaluate_strategy(df, strategy_name, strategy_config)
        all_results.extend(strategy_results)

    if not all_results:
        print("\n❌ No images were evaluated. Please check your reconstruction paths.")
        return

    # Create DataFrame and save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_METRICS_CSV, index=False)

    print("\n" + "=" * 70)
    print("TRANSFORMER PERFORMANCE SUMMARY")
    print("=" * 70)

    # Group by strategy to show average performance
    summary = results_df.groupby("strategy")[["mse", "psnr", "ssim"]].mean()
    print(summary)
    
    print(f"\n📂 Detailed results saved to: {OUTPUT_METRICS_CSV}")
    print("💡 Suggestion: Compare these SSIM values against your Rigid/MLP results.")

if __name__ == "__main__":
    main()