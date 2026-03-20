import os
import numpy as np

# ================= CONFIGURATION =================
# Path to the 'py' folder containing the .npy files

ROI_DATA_PATH = (
    r"D:/PBL_6/DATA_prep/BOLD5000_GLMsingle_ROI_betas/BOLD5000_GLMsingle_ROI_betas/py"
)

OUTPUT_DIR = r"D:/PBL_6/DATA_prep/output/brain_features"
SUBJECT = "CSI1"  # We focus on Subject 1 for now 



def merge_rois():
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"🔍 Searching for {SUBJECT} files in '{ROI_DATA_PATH}'...")

    # These are the specific brain regions we want to combine
    # L/R = Left/Right Hemisphere
    # EarlyVis = Basic lines/shapes (V1, V2, V3)
    # LOC = Objects (Lateral Occipital Complex)
    # PPA = Scenes (Parahippocampal Place Area)
    # OPA/RSC = Navigation/Space
    target_rois = [
        "LHEarlyVis",
        "RHEarlyVis",
        "LHLOC",
        "RHLOC",
        "LHPPA",
        "RHPPA",
        "LHOPA",
        "RHOPA",
        "LHRSC",
        "RHRSC",
    ]

    combined_fmri = []

    for roi in target_rois:
        # Pattern: CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_allses_[ROI].npy
        filename = f"{SUBJECT}_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_allses_{roi}.npy"
        full_path = os.path.join(ROI_DATA_PATH, filename)

        if os.path.exists(full_path):
            print(f"  - Loading {roi}...")
            data = np.load(full_path)

            # The data shape is (Number_of_Images, Number_of_Voxels)
            # We add it to our list to stack them side-by-side
            combined_fmri.append(data)
        else:
            print(f"  ⚠️ Warning: Could not find file for {roi} at {full_path}")

    
    if combined_fmri:
        full_brain_matrix = np.hstack(combined_fmri)

        print(f"\n🧠 Merge Complete for {SUBJECT}!")
        print(f"  - Final Shape: {full_brain_matrix.shape}")
        print(
            f"    (That means {full_brain_matrix.shape[0]} images, {full_brain_matrix.shape[1]} voxels)"
        )

        save_path = os.path.join(OUTPUT_DIR, f"{SUBJECT}_merged_brain.npy")
        np.save(save_path, full_brain_matrix)
        print(f"  - Saved to: {save_path}")
    else:
        print("\n Error: No files were loaded. Check your 'ROI_DATA_PATH'.")


if __name__ == "__main__":
    merge_rois()
