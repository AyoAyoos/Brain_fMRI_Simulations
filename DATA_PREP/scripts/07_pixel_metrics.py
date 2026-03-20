import os

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error

# ================= CONFIGURATION =================
# Path to Real Images (Ground Truth)
IMAGES_DIR = r"D:\PBL_6\DATA_prep\Original_Images"
# Path to Reconstructed Images
RECON_DIR = r"D:\PBL_6\DATA_prep\output\models"
# Path to the CSV Splits (To know which image is which)
SPLITS_DIR = r"D:/PBL_6/DATA_prep/output/splits"
# =================================================


def load_image(path):
    return np.array(Image.open(path).convert("RGB").resize((512, 512)))


def calculate_metrics():
    print("🔍 DIAGNOSTIC MODE: Checking paths...")

    # 1. Check Scene List
    csv_path = os.path.join(SPLITS_DIR, "test_scenes_coco.csv")
    if not os.path.exists(csv_path):
        print(f"❌ CRITICAL ERROR: CSV not found at {os.path.abspath(csv_path)}")
        return
    df = pd.read_csv(csv_path)
    print(f"   - Loaded list of {len(df)} scenes.")

    # 2. Check Recon Folder
    recon_folder = os.path.join(RECON_DIR, "pred_scenes")
    if not os.path.exists(recon_folder):
        print(
            f"❌ CRITICAL ERROR: Recon folder not found at {os.path.abspath(recon_folder)}"
        )
        print("   (Did you name the input file 'pred_scenes.npy' in Step 6?)")
        return

    # Check what is actually inside the folder
    files_in_recon = os.listdir(recon_folder)
    print(f"   - Found {len(files_in_recon)} images in 'pred_scenes' folder.")
    if len(files_in_recon) > 0:
        print(f"   - Example file: {files_in_recon[0]}")

    # 3. Try to process the first few images
    mse_scores = []
    ssim_scores = []

    print("\n📉 Testing first 5 matches...")

    for i, row in df.iterrows():
        # Stop after 5 checks if we are just debugging
        if i >= 5 and len(mse_scores) == 0:
            print("\n❌ STOPPING: First 5 failed. Check the paths above!")
            break

        # Construct Paths
        real_path = os.path.join(IMAGES_DIR, row["folder"], row["filename"])
        recon_path = os.path.join(recon_folder, f"recon_{i:03d}.png")

        # DEBUG: Print the first attempt
        if i == 0:
            print(f"   👉 Attempting to match Pair #0:")
            print(f"      Real:  {os.path.abspath(real_path)}")
            print(f"      Recon: {os.path.abspath(recon_path)}")

        # Check existence
        if not os.path.exists(real_path):
            if i < 3:
                print(f"      ⚠️ Missing REAL image: {row['filename']}")
            continue

        if not os.path.exists(recon_path):
            if i < 3:
                print(f"      ⚠️ Missing RECON image: recon_{i:03d}.png")
            continue

        # Load and Calculate
        try:
            real_img = load_image(real_path)
            recon_img = load_image(recon_path)

            mse = mean_squared_error(real_img.flatten(), recon_img.flatten())
            ss = ssim(real_img, recon_img, channel_axis=2, data_range=255)

            mse_scores.append(mse)
            ssim_scores.append(ss)
            print(f"      ✅ Match found! MSE: {mse:.2f}")

        except Exception as e:
            print(f"      ❌ Error loading image: {e}")

    # 4. Final Stats
    if len(mse_scores) > 0:
        print("\n📝 PIXEL METRICS SUMMARY:")
        print(f"   📊 Average MSE: {np.mean(mse_scores):.2f}")
        print(f"   📊 Average SSIM: {np.mean(ssim_scores):.4f}")
    else:
        print("\n❌ NO MATCHES FOUND. See errors above.")


if __name__ == "__main__":
    calculate_metrics()
