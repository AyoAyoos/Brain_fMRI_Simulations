import os
import re

import numpy as np
import pandas as pd

# ================= CONFIGURATION =================
# Path to the Presentation Lists (The "Schedule")
# Check this path matches your folder structure!
LISTS_DIR = r"D:/PBL_6/DATA_prep/CSI1"

# Path to the Brain Data (Step 2 Output)
BRAIN_DATA_PATH = r"D:/PBL_6/DATA_prep/output/brain_features/CSI1_merged_brain.npy"

# Path to the CSV Splits (Step 1 Output)
SPLITS_DIR = r"D:/PBL_6/DATA_prep/output/splits"

# Output for the aligned brain data
OUTPUT_DIR = r"D:/PBL_6/DATA_prep/output/aligned_brain"
# =================================================


def load_presentation_order():
    """
    Reads all session text files to build a map: Image_Name -> [Row_Indices_in_Brain_Data]
    """
    print("⏳ Parsing presentation lists...")
    image_to_brain_rows = {}
    current_row_idx = 0

    # BOLD5000 CSI1 has 15 sessions. We must read them in order.
    # We iterate 1 to 15 (some subjects have fewer, but CSI1 has 15)
    for sess in range(1, 16):
        # Format: CSI1_sess01, CSI1_sess02...
        sess_name = f"CSI1_sess{sess:02d}"
        sess_path = os.path.join(LISTS_DIR, sess_name)

        if not os.path.exists(sess_path):
            print(f"  ⚠️ Warning: Session folder not found: {sess_name}")
            continue

        # Inside each session, runs are usually 01 to 09 or 10
        # We need to sort them alphabetically to ensure 01 comes before 02
        files = sorted(os.listdir(sess_path))

        for f in files:
            # We only want the text files that list images (run01.txt, etc.)
            if f.endswith(".txt") and "run" in f:
                file_path = os.path.join(sess_path, f)
                with open(file_path, "r") as open_file:
                    content = open_file.readlines()

                for line in content:
                    img_name = line.strip()
                    if not img_name:
                        continue  # skip empty lines

                    if img_name not in image_to_brain_rows:
                        image_to_brain_rows[img_name] = []

                    # Store the index and increment
                    image_to_brain_rows[img_name].append(current_row_idx)
                    current_row_idx += 1

    print(
        f"✅ Mapped {len(image_to_brain_rows)} unique images from {current_row_idx} total trials."
    )
    return image_to_brain_rows


def align_data():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Load the big brain matrix
    print(f"\n🧠 Loading Brain Data: {BRAIN_DATA_PATH}...")
    full_brain_data = np.load(BRAIN_DATA_PATH)
    print(f"   - Shape: {full_brain_data.shape}")

    # 2. Load the map
    img_map = load_presentation_order()

    # 3. Process each split to create matching Brain Data
    splits = ["train_mixed", "test_objects_imagenet", "test_scenes_coco"]

    for split_name in splits:
        csv_path = os.path.join(SPLITS_DIR, f"{split_name}.csv")
        if not os.path.exists(csv_path):
            continue

        print(f"\n🔄 Aligning {split_name}...")
        df = pd.read_csv(csv_path)

        aligned_brain_rows = []
        valid_indices = []

        for i, row in df.iterrows():
            img_name = row["filename"]

            # Remove extension for matching if necessary (sometimes lists have .jpg, sometimes not)
            # But BOLD5000 lists usually include it.

            if img_name in img_map:
                # GET THE BRAIN DATA
                # If an image was shown 3 times, we Average the 3 brain scans (Denoising!)
                indices = img_map[img_name]
                brain_samples = full_brain_data[indices]
                avg_brain = np.mean(brain_samples, axis=0)

                aligned_brain_rows.append(avg_brain)
                valid_indices.append(i)
            else:
                # Try checking without extension if mismatch
                base_name = os.path.splitext(img_name)[0]
                if base_name in img_map:
                    indices = img_map[base_name]
                    brain_samples = full_brain_data[indices]
                    avg_brain = np.mean(brain_samples, axis=0)
                    aligned_brain_rows.append(avg_brain)
                    valid_indices.append(i)
                else:
                    print(f"   ❌ Missing brain data for: {img_name}")

        # Save the new aligned file
        if aligned_brain_rows:
            aligned_array = np.vstack(aligned_brain_rows)
            save_name = f"{split_name}_brain.npy"
            np.save(os.path.join(OUTPUT_DIR, save_name), aligned_array)
            print(f"✅ Saved {aligned_array.shape} to {save_name}")

            # Optional: Save a filtered CSV if we lost any images (so X and Y match lengths)
            # But usually we shouldn't lose many.


if __name__ == "__main__":
    align_data()
