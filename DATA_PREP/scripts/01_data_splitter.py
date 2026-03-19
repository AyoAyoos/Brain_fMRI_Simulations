import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ================= CONFIGURATION =================
# Update this to the path where your "original images" folder is
# Example: ".../BOLD5000_Stimuli/Scene_Stimuli/original images"
BASE_PATH = r"D:/PBL_6/DATA_prep/Original_Images"

OUTPUT_DIR = r"D:/PBL_6/DATA_prep/output/splits"
# =================================================

def get_images_from_folder(folder_name):
    """
    scans a specific subfolder and returns a list of its images
    """
    folder_path = os.path.join(BASE_PATH, folder_name)
    images = []
    
    if not os.path.exists(folder_path):
        print(f"⚠️  Warning: Folder not found: {folder_path}")
        print(f"    (Check if your folder is named '{folder_name}' or something else?)")
        return []

    print(f"Scanning {folder_name}...")
    for f in os.listdir(folder_path):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff')):
            images.append({
                "filename": f,
                "folder": folder_name,
                "full_path": os.path.join(folder_path, f)
            })
    return images

def generate_splits():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Load images from the specific folders
    # Check your actual folder names! usually: 'ImageNet', 'COCO', 'Scene'
    coco_imgs = get_images_from_folder("COCO")

    imagenet_imgs = get_images_from_folder("ImageNet")
    
    
    # Try 'Scene' or 'SUN' (depending on download version)
    scene_imgs = get_images_from_folder("Scene")
    if not scene_imgs:
        scene_imgs = get_images_from_folder("SUN")

    # Convert to DataFrames
    df_obj = pd.DataFrame(imagenet_imgs)  # Objects
    df_scene = pd.DataFrame(coco_imgs)    # Complex Scenes (COCO)
    df_sun = pd.DataFrame(scene_imgs)     # Panoramic Scenes (SUN)

    print(f"\nFound:")
    print(f" - ImageNet (Objects): {len(df_obj)}")
    print(f" - COCO (Complex Scenes): {len(df_scene)}")
    print(f" - Scene/SUN (Backgrounds): {len(df_sun)}")

    # 2. Create the Splits (The "Forensic Audit" Logic)
    
    # Test Set A: 100 Random Objects
    train_obj, test_obj = train_test_split(df_obj, test_size=100, random_state=42)
    
    # Test Set B: 100 Random Complex Scenes
    train_coco, test_scene = train_test_split(df_scene, test_size=100, random_state=42)
    
    # Training Set: EVERYTHING else
    # (Rest of ImageNet + Rest of COCO + All SUN)
    training_set = pd.concat([train_obj, train_coco, df_sun])
    
    # 3. Save
    test_obj.to_csv(os.path.join(OUTPUT_DIR, "test_objects_imagenet.csv"), index=False)
    test_scene.to_csv(os.path.join(OUTPUT_DIR, "test_scenes_coco.csv"), index=False)
    training_set.to_csv(os.path.join(OUTPUT_DIR, "train_mixed.csv"), index=False)
    
    print("\n✅ SUCCESS: Splits created in 'output/splits/'")

if __name__ == "__main__":
    generate_splits()