import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 200
HIDDEN_DIM = 2048
DROPOUT = 0.4
WEIGHT_DECAY = 0.05
TEMPERATURE = 0.05
PATIENCE = 20

# Output Filename Prefix
FILE_PREFIX = "rigid_new"

# ==========================================
# PATH CONFIGURATION (Fixed for your Directory)
# ==========================================
BASE_DIR = "D:/PBL_6/DATA_prep/output"
BRAIN_DIR = f"{BASE_DIR}/aligned_brain"
CLIP_DIR  = f"{BASE_DIR}/clip_features"
MODELS_DIR = f"{BASE_DIR}/models"

os.makedirs(MODELS_DIR, exist_ok=True)

# ==========================================
# MODEL ARCHITECTURE (Rigid Optimized)
# ==========================================
class RigidNewEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            
            nn.Linear(HIDDEN_DIM, output_dim)
        )

    def forward(self, x):
        x = self.model(x)
        # Ensure L2 normalization for CLIP space compatibility
        return nn.functional.normalize(x, p=2, dim=1)

# ==========================================
# LOSS UTILITIES
# ==========================================
def contrastive_loss(predictions, targets, temp=0.07):
    logits = torch.matmul(predictions, targets.T) / temp
    labels = torch.arange(predictions.size(0)).to(predictions.device)
    return nn.CrossEntropyLoss()(logits, labels)

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Rigid Model Training on: {device}")

    # --- Path Validation ---
    for path in [BRAIN_DIR, CLIP_DIR]:
        if not os.path.exists(path):
            print(f"❌ ERROR: Directory not found: {path}")
            return

    # 1. Load Data from specific subfolders
    print("📂 Loading data...")
    try:
        train_brain = np.load(f"{BRAIN_DIR}/train_mixed_brain.npy")
        train_clip = np.load(f"{CLIP_DIR}/train_mixed_clip_embeds.npy")
        test_obj_brain = np.load(f"{BRAIN_DIR}/test_objects_imagenet_brain.npy")
        test_scn_brain = np.load(f"{BRAIN_DIR}/test_scenes_coco_brain.npy")
        print(f"✅ Data loaded. Train samples: {train_brain.shape[0]}")
    except FileNotFoundError as e:
        print(f"❌ ERROR: Could not find .npy files. {e}")
        return

    # 2. Preprocessing
    scaler = StandardScaler()
    train_brain_scaled = scaler.fit_transform(train_brain)
    test_obj_scaled = scaler.transform(test_obj_brain)
    test_scn_scaled = scaler.transform(test_scn_brain)

    # 3. Split (80/20)
    x_train, x_val, y_train, y_val = train_test_split(
        train_brain_scaled, train_clip, test_size=0.2, random_state=42
    )
    
    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train).float(), torch.tensor(y_train).float()), 
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x_val).float(), torch.tensor(y_val).float()), 
        batch_size=BATCH_SIZE
    )

    # 4. Initialize Model & Optimizer
    model = RigidNewEncoder(train_brain.shape[1], train_clip.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = []

    # 5. Training Loop
    print("\n" + "="*40)
    print("STARTING TRAINING")
    print("="*40)
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            
            # Hybrid Loss
            loss_mse = nn.MSELoss()(pred, y)
            loss_cos = 1 - torch.mean(nn.functional.cosine_similarity(pred, y))
            loss_cont = contrastive_loss(pred, y, temp=TEMPERATURE)
            
            loss = (1.0 * loss_mse) + (1.0 * loss_cos) + (0.5 * loss_cont)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                total_val_loss += nn.MSELoss()(pred, y).item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        history.append({"epoch": epoch, "val_loss": avg_val_loss})

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train Loss: {total_train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.6f}")

        # Save Best Model with prefix
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{MODELS_DIR}/{FILE_PREFIX}_model_best.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⏹️ Early stopping triggered at epoch {epoch}.")
                break

    # 6. Save Artifacts
    with open(f"{MODELS_DIR}/{FILE_PREFIX}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    pd.DataFrame(history).to_csv(f"{MODELS_DIR}/{FILE_PREFIX}_training_log.csv", index=False)

    # 7. Final Evaluation & Predictions
    print("\n✅ Training Complete. Generating Predictions for Step 06...")
    model.load_state_dict(torch.load(f"{MODELS_DIR}/{FILE_PREFIX}_model_best.pt"))
    model.eval()
    
    with torch.no_grad():
        pred_obj = model(torch.tensor(test_obj_scaled).float().to(device)).cpu().numpy()
        pred_scn = model(torch.tensor(test_scn_scaled).float().to(device)).cpu().numpy()
        
        np.save(f"{MODELS_DIR}/{FILE_PREFIX}_pred_objects.npy", pred_obj)
        np.save(f"{MODELS_DIR}/{FILE_PREFIX}_pred_scenes.npy", pred_scn)

    print("\n" + "="*40)
    print(f"SUCCESS: Artifacts saved with prefix '{FILE_PREFIX}'")
    print(f"Location: {MODELS_DIR}")
    print("="*40)

if __name__ == "__main__":
    main()