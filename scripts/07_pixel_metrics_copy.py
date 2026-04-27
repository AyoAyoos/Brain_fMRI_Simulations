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

# Ground-truth images
IMAGES_DIR = os.path.join(BASE_DIR, "Original_Images")

# CSV split used by 06c
SPLITS_DIR = os.path.join(BASE_DIR, "output", "splits")
SPLIT_CSV = "test_scenes_coco.csv"

# 06c output folder
RECON_BASE_DIR = os.path.join(BASE_DIR, "output", "reconstructions")

# This must match 06c:
# INPUT_FILE = "pred_scenes_mlp.npy"
# run_label = INPUT_FILE.replace(".npy", "") + "_sdxl"
RUN_LABEL = "pred_scenes_mlp_sdxl"

# SDXL output resolution
IMAGE_SIZE = 1024

# Choose which strategies to evaluate
STRATEGIES = {
    "direct": {
        "folder": "direct",
        "filename_pattern": "recon_direct_{:03d}.png",
    },
    "retrieval": {
        "folder": "retrieval",
        "filename_pattern": "recon_retrieval_{:03d}.png",
    },
}

# Save detailed metrics
OUTPUT_METRICS_CSV = os.path.join(
    BASE_DIR, "output", "reconstructions", RUN_LABEL, "metrics_summary.csv"
)


# ============================================================
# IMAGE HELPERS
# ============================================================

def load_image(path, size=1024):
    """
    Loads image as RGB and resizes to target size.
    """
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size))
    return np.array(img)


def calculate_psnr(mse_value, max_pixel=255.0):
    """
    Calculates PSNR from MSE.
    Higher PSNR = better reconstruction.
    """
    if mse_value == 0:
        return float("inf")
    return 20 * math.log10(max_pixel / math.sqrt(mse_value))


def safe_mean(values):
    return float(np.mean(values)) if len(values) > 0 else None


# ============================================================
# METRIC CALCULATION
# ============================================================

def evaluate_strategy(df, strategy_name, strategy_config):
    """
    Evaluates one reconstruction strategy:
    - direct
    - retrieval
    """
    recon_folder = os.path.join(
        RECON_BASE_DIR,
        RUN_LABEL,
        strategy_config["folder"]
    )

    print("\n" + "=" * 70)
    print(f"Evaluating Strategy: {strategy_name.upper()}")
    print("=" * 70)
    print(f"Recon folder: {recon_folder}")

    if not os.path.exists(recon_folder):
        print(f"❌ Folder not found: {recon_folder}")
        return []

    files = os.listdir(recon_folder)
    print(f"Found {len(files)} files in reconstruction folder.")

    results = []

    for i, row in df.iterrows():
        real_path = os.path.join(IMAGES_DIR, row["folder"], row["filename"])
        recon_filename = strategy_config["filename_pattern"].format(i)
        recon_path = os.path.join(recon_folder, recon_filename)

        if i == 0:
            print("\nFirst pair check:")
            print(f"Real image : {real_path}")
            print(f"Recon image: {recon_path}")

        if not os.path.exists(real_path):
            print(f"⚠️ Missing real image for index {i}: {real_path}")
            continue

        if not os.path.exists(recon_path):
            # Stop printing too much noise after first few missing files
            if i < 5:
                print(f"⚠️ Missing reconstructed image for index {i}: {recon_path}")
            continue

        try:
            real_img = load_image(real_path, IMAGE_SIZE)
            recon_img = load_image(recon_path, IMAGE_SIZE)

            mse_value = mean_squared_error(real_img.flatten(), recon_img.flatten())
            ssim_value = ssim(
                real_img,
                recon_img,
                channel_axis=2,
                data_range=255
            )
            psnr_value = calculate_psnr(mse_value)

            results.append({
                "strategy": strategy_name,
                "index": i,
                "filename": row["filename"],
                "folder": row["folder"],
                "real_path": real_path,
                "recon_path": recon_path,
                "mse": mse_value,
                "ssim": ssim_value,
                "psnr": psnr_value,
            })

            if i < 5:
                print(
                    f"✅ Index {i}: "
                    f"MSE={mse_value:.2f}, "
                    f"SSIM={ssim_value:.4f}, "
                    f"PSNR={psnr_value:.2f}"
                )

        except Exception as e:
            print(f"❌ Error at index {i}: {e}")

    return results


def calculate_metrics():
    print("🔍 Checking SDXL reconstruction metrics setup...")

    csv_path = os.path.join(SPLITS_DIR, SPLIT_CSV)

    if not os.path.exists(csv_path):
        print(f"❌ Split CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded split CSV: {csv_path}")
    print(f"Total ground-truth images listed: {len(df)}")

    all_results = []

    for strategy_name, strategy_config in STRATEGIES.items():
        strategy_results = evaluate_strategy(df, strategy_name, strategy_config)
        all_results.extend(strategy_results)

    if len(all_results) == 0:
        print("\n❌ No valid image pairs found. Check folders and filenames.")
        return

    results_df = pd.DataFrame(all_results)

    os.makedirs(os.path.dirname(OUTPUT_METRICS_CSV), exist_ok=True)
    results_df.to_csv(OUTPUT_METRICS_CSV, index=False)

    print("\n" + "=" * 70)
    print("FINAL METRICS SUMMARY")
    print("=" * 70)

    for strategy_name in results_df["strategy"].unique():
        sub = results_df[results_df["strategy"] == strategy_name]

        print(f"\nStrategy: {strategy_name.upper()}")
        print(f"Images evaluated: {len(sub)}")
        print(f"Average MSE : {sub['mse'].mean():.2f}")
        print(f"Average SSIM: {sub['ssim'].mean():.4f}")
        print(f"Average PSNR: {sub['psnr'].mean():.2f}")

    print("\n✅ Detailed metrics saved to:")
    print(OUTPUT_METRICS_CSV)


if __name__ == "__main__":
    calculate_metrics()