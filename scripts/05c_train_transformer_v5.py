"""
05b_train_transformer_v5.py
===========================
PHASE 5: ROI-Aware Brain Transformer (FIXED)
Treats the 10 merged ROIs as a sequence of tokens. 
Uses a Transformer Encoder to capture cross-region dynamics.

FIXED: Resolved mat1/mat2 shape mismatch by aligning padding and chunk_size.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BRAIN_DIR  = r"D:/PBL_6/DATA_prep/output/aligned_brain"
CLIP_DIR   = r"D:/PBL_6/DATA_prep/output/clip_features"
MODELS_DIR = r"D:/PBL_6/DATA_prep/output/models"

FILE_PREFIX = "mlp_v5_transformer"

# ROI Architecture
# [LHEarlyVis, RHEarlyVis, LHLOC, RHLOC, LHPPA, RHPPA, LHOPA, RHOPA, LHRSC, RHRSC]
NUM_ROIS = 10  

# Hyperparameters
BATCH_SIZE    = 32
LEARNING_RATE = 1e-4    
EPOCHS        = 100     
HIDDEN_DIM    = 512    
CLIP_DIM      = 1024   
DROPOUT       = 0.3    
TEMPERATURE   = 0.05   
CONTRASTIVE_MAX_WT = 2.0 

os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL: Brain-Transformer
# ─────────────────────────────────────────────────────────────────────────────
class BrainTransformer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 1. Calculate proper chunking with padding
        # We find the smallest multiple of NUM_ROIS >= input_dim
        self.padded_dim = int(np.ceil(input_dim / NUM_ROIS) * NUM_ROIS)
        self.chunk_size = self.padded_dim // NUM_ROIS
        
        # 2. ROI Projection
        self.roi_embed = nn.Linear(self.chunk_size, HIDDEN_DIM)
        
        # 3. Positional Encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, NUM_ROIS, HIDDEN_DIM))
        
        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM, 
            nhead=8, 
            dim_feedforward=HIDDEN_DIM * 2, 
            dropout=DROPOUT,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # 5. Output Heads
        self.ln_final = nn.LayerNorm(HIDDEN_DIM)
        self.out_proj = nn.Linear(HIDDEN_DIM, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Correctly pad to the expected padded_dim (e.g., 1685 -> 1690)
        pad_needed = self.padded_dim - x.shape[1]
        if pad_needed > 0:
            x = F.pad(x, (0, pad_needed))
        
        # Reshape into (batch, 10, chunk_size)
        # This will now be exactly (batch, 10, 169)
        tokens = x.view(batch_size, NUM_ROIS, self.chunk_size)
        
        # Project tokens to hidden space
        x = self.roi_embed(tokens) # (batch, 10, HIDDEN_DIM)
        x = x + self.pos_embedding
        
        # Process through Transformer
        x = self.transformer(x)
        
        # Average pooling across ROI tokens
        x = x.mean(dim=1)
        x = self.ln_final(x)
        x = self.out_proj(x)
        
        return F.normalize(x, p=2, dim=-1)

# ─────────────────────────────────────────────────────────────────────────────
# LOSS & TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def contrastive_loss(pred, target, temp=0.05):
    logits = torch.matmul(pred, target.T) / temp
    labels = torch.arange(pred.size(0)).to(pred.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Phase 5: Training Brain-Transformer on: {device}")

    # Load Data
    X = np.load(os.path.join(BRAIN_DIR, "train_mixed_brain.npy"))
    Y = np.load(os.path.join(CLIP_DIR, "train_mixed_clip_embeds.npy"))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    x_train, x_val, y_train, y_val = train_test_split(X_scaled, Y, test_size=0.15, random_state=42)
    
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).float()), batch_size=BATCH_SIZE)

    model = BrainTransformer(X.shape[1], CLIP_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    best_loss = float('inf')
    history = []
    best_model_path = os.path.join(MODELS_DIR, f"{FILE_PREFIX}_best.pt")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        
        # Smooth Contrastive Warmup
        current_cont_wt = min(CONTRASTIVE_MAX_WT, (epoch / 20.0) * CONTRASTIVE_MAX_WT)
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            
            l_mse  = F.mse_loss(pred, y)
            l_cos  = 1 - torch.mean(F.cosine_similarity(pred, y))
            l_cont = contrastive_loss(pred, y, temp=TEMPERATURE)
            
            loss = (5.0 * l_mse) + (2.0 * l_cos) + (current_cont_wt * l_cont)
            loss.backward()
            optimizer.step()

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                v_loss += (1 - torch.mean(F.cosine_similarity(pred, y))).item()
        
        avg_v_loss = v_loss / len(val_loader)
        history.append({"epoch": epoch, "val_loss": avg_v_loss})
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Cont_Wt: {current_cont_wt:.2f} | Val Loss: {avg_v_loss:.4f}")

        if avg_v_loss < best_loss:
            best_loss = avg_v_loss
            torch.save(model.state_dict(), best_model_path)

    # Save artifacts
    joblib.dump(scaler, os.path.join(MODELS_DIR, f"{FILE_PREFIX}_scaler.pkl"))
    pd.DataFrame(history).to_csv(os.path.join(MODELS_DIR, f"{FILE_PREFIX}_training_log.csv"), index=False)
    
    # Generate predictions
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    for test_name in ["test_objects_imagenet", "test_scenes_coco"]:
        try:
            x_test_raw = np.load(os.path.join(BRAIN_DIR, f"{test_name}_brain.npy"))
            x_test_scaled = scaler.transform(x_test_raw)
            with torch.no_grad():
                pred = model(torch.from_numpy(x_test_scaled).float().to(device)).cpu().numpy()
            np.save(os.path.join(MODELS_DIR, f"{FILE_PREFIX}_{test_name}_preds.npy"), pred)
            print(f"✅ Generated predictions for {test_name}")
        except: pass

    print(f"\n✅ Phase 5 Transformer complete. File Prefix: {FILE_PREFIX}")

if __name__ == "__main__":
    main()