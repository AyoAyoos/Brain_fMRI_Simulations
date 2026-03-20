import os

import joblib
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ================= CONFIGURATION =================
# Paths to your processed data
BRAIN_DIR = r"D:/PBL_6/DATA_prep/output/aligned_brain"
CLIP_DIR = r"D:/PBL_6/DATA_prep/output/clip_features"
MODELS_DIR = r"D:/PBL_6/DATA_prep/output/models"

# The alpha values (regularization strength) to test.
# fMRI usually likes high alphas (1000 to 60000) because the data is noisy.
ALPHAS = [100, 1000, 10000, 50000]
# =================================================


def evaluate_predictions(y_true, y_pred, label):
    """
    Calculates the correlation between the True CLIP vector and the Predicted one.
    This is your "Accuracy Score".
    """
    correlations = []
    # Calculate correlation for each image (row-by-row)
    for i in range(len(y_true)):
        corr, _ = pearsonr(y_true[i], y_pred[i])
        correlations.append(corr)

    avg_score = np.mean(correlations)
    print(f"   📊 {label} Score: {avg_score:.4f} (Max: {np.max(correlations):.4f})")
    return avg_score


def train_and_test():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    # 1. LOAD TRAINING DATA
    print("⏳ Loading Training Data...")
    try:
        X_train = np.load(os.path.join(BRAIN_DIR, "train_mixed_brain.npy"))
        Y_train = np.load(os.path.join(CLIP_DIR, "train_mixed_clip_embeds.npy"))

        # Double check shapes match
        if X_train.shape[0] != Y_train.shape[0]:
            # Sometimes alignment drops a few frames. We trim to the smaller size.
            min_len = min(X_train.shape[0], Y_train.shape[0])
            X_train = X_train[:min_len]
            Y_train = Y_train[:min_len]
            print(f"   ⚠️ Trimmed data to {min_len} samples to match.")

        print(f"   - Training on {len(X_train)} samples.")

    except FileNotFoundError:
        print("❌ Error: Training files not found. Did you run Step 4?")
        return

    # 2. TRAIN THE MODEL
    print("\n🧠 Training Ridge Regression (This acts as the 'Mind Reader')...")
    # We use a Pipeline: First standardize the brain data (Z-score), then run Ridge
    model = make_pipeline(
        StandardScaler(), RidgeCV(alphas=ALPHAS, scoring="neg_mean_squared_error")
    )

    model.fit(X_train, Y_train)

    # Save the trained model
    save_path = os.path.join(MODELS_DIR, "csi1_brain_translator.pkl")
    joblib.dump(model, save_path)
    print(f"✅ Model trained and saved to {save_path}")

    # 3. THE FORENSIC AUDIT (Testing)
    print("\n🔎 RUNNING THE AUDIT (Object vs. Scene)...")

    # --- Test A: OBJECTS ---
    try:
        X_obj = np.load(os.path.join(BRAIN_DIR, "test_objects_imagenet_brain.npy"))
        Y_obj = np.load(os.path.join(CLIP_DIR, "test_objects_imagenet_clip_embeds.npy"))

        # Trim if needed
        min_len = min(X_obj.shape[0], Y_obj.shape[0])
        X_obj, Y_obj = X_obj[:min_len], Y_obj[:min_len]

        # Predict
        Y_pred_obj = model.predict(X_obj)
        score_obj = evaluate_predictions(Y_obj, Y_pred_obj, "Simple Objects (ImageNet)")

        # Save predictions for later image generation
        np.save(os.path.join(MODELS_DIR, "pred_objects.npy"), Y_pred_obj)

    except FileNotFoundError:
        print("   ⚠️ Object test files missing.")

    # --- Test B: SCENES ---
    try:
        X_scene = np.load(os.path.join(BRAIN_DIR, "test_scenes_coco_brain.npy"))
        Y_scene = np.load(os.path.join(CLIP_DIR, "test_scenes_coco_clip_embeds.npy"))

        # Trim if needed
        min_len = min(X_scene.shape[0], Y_scene.shape[0])
        X_scene, Y_scene = X_scene[:min_len], Y_scene[:min_len]

        # Predict
        Y_pred_scene = model.predict(X_scene)
        score_scene = evaluate_predictions(
            Y_scene, Y_pred_scene, "Complex Scenes (COCO)"
        )

        # Save predictions
        np.save(os.path.join(MODELS_DIR, "pred_scenes.npy"), Y_pred_scene)

    except FileNotFoundError:
        print("   ⚠️ Scene test files missing.")

    # 4. FINAL VERDICT
    print("\n📝 RESULTS SUMMARY:")
    if "score_obj" in locals() and "score_scene" in locals():
        diff = score_obj - score_scene
        print(f"   Gap: {diff:.4f}")
        if diff > 0:
            print(
                "   ✅ HYPOTHESIS CONFIRMED: The model understands Objects better than Scenes."
            )
        else:
            print(
                "   ❓ RESULT: Scores are similar (or Scenes are better). Interesting!"
            )


if __name__ == "__main__":
    train_and_test()
