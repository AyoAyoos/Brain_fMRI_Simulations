"""
05b_train_mlp_translator.py (IMPROVED VERSION)
===============================================
Enhanced MLP translator with stronger architecture and comprehensive diagnostics.

MAJOR IMPROVEMENTS FROM PREVIOUS VERSION:
1. Train/validation split with proper evaluation
2. L2 normalization of CLIP embeddings (prevents collapse)
3. Residual MLP architecture (stronger feature learning)
4. Combined loss: MSE + Cosine + Symmetric InfoNCE contrastive
5. Early stopping + ReduceLROnPlateau scheduler
6. Retrieval accuracy metrics (top-1, top-5)
7. Comprehensive sanity checks and diagnostics
8. Save best model based on validation loss

HOW TO RUN:
  python 05b_train_mlp_translator.py

REQUIREMENTS:
  pip install torch numpy scikit-learn scipy joblib pandas
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
from scipy.stats import pearsonr
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BRAIN_DIR  = r"D:/PBL_6/DATA_prep/output/aligned_brain"
CLIP_DIR   = r"D:/PBL_6/DATA_prep/output/clip_features"
MODELS_DIR = r"D:/PBL_6/DATA_prep/output/models"

# Training hyperparameters
EPOCHS         = 200      # More epochs with early stopping
BATCH_SIZE     = 64       # Will adjust based on dataset size
LEARNING_RATE  = 1e-4     # Lower LR for stability
WEIGHT_DECAY   = 1e-2     # L2 regularization
DROPOUT_1      = 0.4     # First dropout rate
DROPOUT_2      = 0.4     # Second dropout rate
TEMPERATURE    = 0.02     # Contrastive temperature (lower = harder negatives)
CLIP_DIM       = 768      # CLIP ViT-L/14 embedding dimension
HIDDEN_DIM     = 2048     # First hidden layer dimension

# Loss weights
WEIGHT_MSE     = 0.5      # MSE loss weight
WEIGHT_COSINE  = 1.0      # Cosine loss weight
WEIGHT_CONTRAST = 1.0     # Contrastive loss weight

# Early stopping & LR scheduling
PATIENCE       = 25       # Early stopping patience
LR_PATIENCE    = 10       # ReduceLROnPlateau patience
LR_FACTOR      = 0.5      # LR reduction factor

# Validation split
VAL_SPLIT      = 0.2      # 20% validation
RANDOM_STATE   = 42       # For reproducibility


# ─────────────────────────────────────────────────────────────────────────────
# 1. RESIDUAL BLOCK
# ─────────────────────────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.
    
    Architecture:
        x → Linear → LayerNorm → GELU → Dropout → Linear → LayerNorm → (+x) → out
    
    Why residual connections?
    - Prevents gradient vanishing in deep networks
    - Allows learning identity mapping when needed
    - Improves training stability for fMRI → CLIP mapping
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.ln2(out)
        # Skip connection - adds input to output
        return out + residual


# ─────────────────────────────────────────────────────────────────────────────
# 2. IMPROVED BRAIN ENCODER WITH RESIDUAL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
class ImprovedBrainEncoder(nn.Module):
    """
    Enhanced MLP with residual connections for stronger feature learning.
    
    Architecture:
        brain_dim → Linear(4096) → LayerNorm → GELU → Dropout(0.25)
                  → ResidualBlock × 2 (4096 dim, dropout 0.2)
                  → Linear(2048) → LayerNorm → GELU → Dropout(0.20)
                  → Linear(CLIP_DIM)
    
    Why this architecture?
    - Initial expansion to 4096 captures complex brain patterns
    - Residual blocks prevent gradient issues and enable deeper learning
    - Gradual compression (4096 → 2048 → 768) preserves information
    - LayerNorm provides stable training on small batches
    """
    def __init__(self, brain_dim: int, clip_dim: int = CLIP_DIM):
        super().__init__()
        
        # Initial expansion layer
        self.input_layer = nn.Sequential(
            nn.Linear(brain_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT_1)
        )
        
        # Residual blocks for deep feature learning
        self.residual_blocks = nn.Sequential(
            ResidualBlock(HIDDEN_DIM, dropout=0.2),
            ResidualBlock(HIDDEN_DIM, dropout=0.2)
        )
        
        # Compression layers
        self.compression = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(DROPOUT_2)
        )
        
        # Final projection to CLIP space (no activation)
        self.output_layer = nn.Linear(2048, clip_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.residual_blocks(x)
        x = self.compression(x)
        x = self.output_layer(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean Squared Error - ensures predictions are close in Euclidean space.
    """
    return F.mse_loss(pred, target)


def cosine_embedding_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Cosine embedding loss: 1 - mean(cosine_similarity).
    
    Why this helps:
    - Encourages angular alignment between predicted and true embeddings
    - Less sensitive to magnitude than MSE
    - CLIP embeddings are typically normalized, so angular distance matters more
    """
    # L2 normalize both (critical for cosine similarity)
    pred_norm = F.normalize(pred, dim=-1, p=2)
    target_norm = F.normalize(target, dim=-1, p=2)
    
    # Cosine similarity: dot product of normalized vectors
    cos_sim = (pred_norm * target_norm).sum(dim=-1)
    
    # Loss = 1 - mean cosine similarity (minimize distance)
    return 1.0 - cos_sim.mean()


def symmetric_contrastive_loss(pred: torch.Tensor, target: torch.Tensor,
                               temperature: float = 0.05) -> torch.Tensor:
    """
    Symmetric InfoNCE contrastive loss (CLIP-style).
    
    For batch of N pairs:
    - pred[i] should match target[i] (positive pair)
    - pred[i] should NOT match target[j≠i] (negative pairs)
    
    Why symmetric?
    - Both pred→target and target→pred directions matter
    - Prevents mode collapse where all predictions are similar
    - Forces model to distinguish between different brain states
    
    Why this prevents embedding collapse:
    - If all predictions collapse to average, they'll match ALL targets equally
    - Contrastive loss penalizes this heavily - forces distinctiveness
    """
    if pred.shape[0] < 2:
        return torch.tensor(0.0, device=pred.device)
    
    # L2 normalize for cosine similarity
    pred_norm = F.normalize(pred, dim=-1, p=2)
    target_norm = F.normalize(target, dim=-1, p=2)
    
    # Similarity matrix: (N, N)
    # logits[i,j] = similarity between pred[i] and target[j]
    logits = torch.matmul(pred_norm, target_norm.T) / temperature
    
    # Labels: diagonal entries are correct pairs
    labels = torch.arange(pred.shape[0], device=pred.device)
    
    # Symmetric cross-entropy: pred→target AND target→pred
    loss_pred_to_target = F.cross_entropy(logits, labels)
    loss_target_to_pred = F.cross_entropy(logits.T, labels)
    
    return (loss_pred_to_target + loss_target_to_pred) / 2.0


def combined_loss(pred: torch.Tensor, target: torch.Tensor,
                 temperature: float = TEMPERATURE) -> tuple:
    """
    Combined loss = 0.5*MSE + 1.0*Cosine + 1.0*Contrastive
    
    Why combine three losses?
    - MSE: Encourages proximity in Euclidean space
    - Cosine: Encourages angular alignment (direction matters)
    - Contrastive: Prevents collapse, enforces distinctiveness
    
    This combination addresses the "noisy abstract textures" problem by:
    1. Preventing all predictions from averaging to the same embedding
    2. Maintaining semantic distinctions between different brain states
    3. Learning both magnitude and direction properly
    """
    l_mse = mse_loss(pred, target)
    l_cosine = cosine_embedding_loss(pred, target)
    l_contrastive = symmetric_contrastive_loss(pred, target, temperature)
    
    # Total weighted loss
    total = (WEIGHT_MSE * l_mse + 
             WEIGHT_COSINE * l_cosine + 
             WEIGHT_CONTRAST * l_contrastive)
    
    return total, l_mse.item(), l_cosine.item(), l_contrastive.item()


# ─────────────────────────────────────────────────────────────────────────────
# 4. EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_cosine_similarity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Average cosine similarity between true and predicted embeddings.
    
    Why this metric?
    - CLIP embeddings are normalized, so cosine similarity is natural
    - Ranges from -1 to 1, with 1 being perfect alignment
    - More interpretable than MSE for normalized vectors
    """
    # L2 normalize
    y_true_norm = y_true / (np.linalg.norm(y_true, axis=1, keepdims=True) + 1e-8)
    y_pred_norm = y_pred / (np.linalg.norm(y_pred, axis=1, keepdims=True) + 1e-8)
    
    # Cosine similarity per sample
    cos_sims = (y_true_norm * y_pred_norm).sum(axis=1)
    return float(np.mean(cos_sims))


def compute_pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Average per-sample Pearson correlation.
    
    Measures linear relationship between true and predicted embeddings.
    """
    correlations = []
    for i in range(len(y_true)):
        corr, _ = pearsonr(y_true[i], y_pred[i])
        if not np.isnan(corr):
            correlations.append(corr)
    return float(np.mean(correlations)) if correlations else 0.0


def compute_retrieval_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                              k_values: list = [1, 5]) -> dict:
    """
    Retrieval accuracy: Can we retrieve the correct image from predictions?
    
    For each predicted embedding:
    1. Compute cosine similarity to ALL true embeddings
    2. Check if the matching true embedding ranks in top-k
    
    Why this metric is critical:
    - Directly measures if predictions preserve semantic identity
    - If embeddings collapsed, all predictions would retrieve random images
    - Top-1 accuracy shows exact matching ability
    - Top-5 accuracy shows semantic neighborhood preservation
    
    This is the BEST indicator of whether Step 06 image generation will work.
    """
    # L2 normalize
    y_true_norm = y_true / (np.linalg.norm(y_true, axis=1, keepdims=True) + 1e-8)
    y_pred_norm = y_pred / (np.linalg.norm(y_pred, axis=1, keepdims=True) + 1e-8)
    
    # Similarity matrix: (N_pred, N_true)
    # sim_matrix[i, j] = similarity between pred[i] and true[j]
    sim_matrix = np.matmul(y_pred_norm, y_true_norm.T)
    
    results = {}
    for k in k_values:
        # For each prediction, get top-k most similar true embeddings
        top_k_indices = np.argsort(sim_matrix, axis=1)[:, -k:]
        
        # Check if correct index (diagonal) is in top-k
        correct = 0
        for i in range(len(y_pred)):
            if i in top_k_indices[i]:
                correct += 1
        
        accuracy = correct / len(y_pred)
        results[f'top{k}'] = accuracy
    
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. DATA PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def sanity_check_data(X: np.ndarray, Y: np.ndarray, name: str):
    """
    Pre-training sanity checks to catch data issues early.
    
    Checks:
    - Shape compatibility
    - NaN/Inf values
    - Data statistics (mean, std)
    - Embedding norms
    """
    print(f"\n{'='*60}")
    print(f"SANITY CHECKS: {name}")
    print(f"{'='*60}")
    
    # Shape check
    print(f"Brain data shape:      {X.shape}")
    print(f"CLIP embeddings shape: {Y.shape}")
    
    if len(X) != len(Y):
        print(f"⚠️  WARNING: Length mismatch! Brain: {len(X)}, CLIP: {len(Y)}")
        min_len = min(len(X), len(Y))
        print(f"   Trimming both to {min_len} samples...")
        X = X[:min_len]
        Y = Y[:min_len]
    else:
        print(f"✅ Sample count matches: {len(X)}")
    
    # NaN/Inf check
    brain_nan = np.isnan(X).any()
    brain_inf = np.isinf(X).any()
    clip_nan = np.isnan(Y).any()
    clip_inf = np.isinf(Y).any()
    
    if brain_nan or brain_inf:
        print(f"❌ Brain data has NaN: {brain_nan}, Inf: {brain_inf}")
        raise ValueError("Invalid values in brain data!")
    if clip_nan or clip_inf:
        print(f"❌ CLIP data has NaN: {clip_nan}, Inf: {clip_inf}")
        raise ValueError("Invalid values in CLIP data!")
    
    print(f"✅ No NaN/Inf values detected")
    
    # Statistics
    print(f"\nBrain data statistics (before scaling):")
    print(f"  Mean: {X.mean():.6f}")
    print(f"  Std:  {X.std():.6f}")
    print(f"  Min:  {X.min():.6f}")
    print(f"  Max:  {X.max():.6f}")
    
    print(f"\nCLIP embeddings statistics (before normalization):")
    print(f"  Mean: {Y.mean():.6f}")
    print(f"  Std:  {Y.std():.6f}")
    print(f"  Mean L2 norm: {np.linalg.norm(Y, axis=1).mean():.6f}")
    print(f"  Std L2 norm:  {np.linalg.norm(Y, axis=1).std():.6f}")
    
    return X, Y


def preprocess_data(X_train_raw: np.ndarray, Y_train_raw: np.ndarray,
                   X_val_raw: np.ndarray, Y_val_raw: np.ndarray) -> tuple:
    """
    Preprocess brain and CLIP data.
    
    Brain data:
    - StandardScaler normalization (z-score)
    - Fit on training, transform both train/val
    
    CLIP embeddings:
    - L2 normalization (critical to prevent collapse)
    - Apply to both train/val
    
    Why L2 normalize CLIP embeddings?
    - CLIP naturally produces normalized embeddings
    - Keeps predictions in same scale
    - Prevents magnitude collapse (all predictions same norm)
    - Makes cosine similarity and dot product equivalent
    """
    print(f"\n{'='*60}")
    print("PREPROCESSING")
    print(f"{'='*60}")
    
    # Brain data: StandardScaler (z-score normalization)
    print("Scaling brain data...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)
    
    print(f"  After scaling - Train mean: {X_train.mean():.6f}, std: {X_train.std():.6f}")
    print(f"  After scaling - Val mean:   {X_val.mean():.6f}, std: {X_val.std():.6f}")
    
    # CLIP embeddings: L2 normalization
    print("\nL2-normalizing CLIP embeddings...")
    Y_train = Y_train_raw / (np.linalg.norm(Y_train_raw, axis=1, keepdims=True) + 1e-8)
    Y_val = Y_val_raw / (np.linalg.norm(Y_val_raw, axis=1, keepdims=True) + 1e-8)
    Y_train = Y_train.astype(np.float32)
    Y_val = Y_val.astype(np.float32)
    
    print(f"  After L2 norm - Train mean norm: {np.linalg.norm(Y_train, axis=1).mean():.6f}")
    print(f"  After L2 norm - Val mean norm:   {np.linalg.norm(Y_val, axis=1).mean():.6f}")
    
    return X_train, Y_train, X_val, Y_val, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 6. VALIDATION EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model: ImprovedBrainEncoder, val_loader: DataLoader,
            device: torch.device) -> tuple:
    """
    Comprehensive validation evaluation.
    
    Returns:
    - Average loss
    - Cosine similarity
    - Pearson r
    - Top-1 retrieval accuracy
    - Top-5 retrieval accuracy
    """
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_targets = []
    n_batches = 0
    
    for x_batch, y_batch in val_loader:
        pred = model(x_batch)
        
        # L2 normalize predictions (critical!)
        pred_norm = F.normalize(pred, dim=-1, p=2)
        
        loss, _, _, _ = combined_loss(pred_norm, y_batch, TEMPERATURE)
        total_loss += loss.item()
        
        # Collect for metrics
        all_preds.append(pred_norm.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())
        n_batches += 1
    
    # Concatenate all batches
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Compute metrics
    avg_loss = total_loss / n_batches
    cos_sim = compute_cosine_similarity(all_targets, all_preds)
    pearson = compute_pearson_r(all_targets, all_preds)
    retrieval = compute_retrieval_accuracy(all_targets, all_preds, k_values=[1, 5])
    
    return avg_loss, cos_sim, pearson, retrieval['top1'], retrieval['top5']


# ─────────────────────────────────────────────────────────────────────────────
# 7. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def train_model(X_train: np.ndarray, Y_train: np.ndarray,
               X_val: np.ndarray, Y_val: np.ndarray,
               device: torch.device) -> tuple:
    """
    Train the improved brain encoder with comprehensive monitoring.
    
    Features:
    - Early stopping (patience=25)
    - ReduceLROnPlateau scheduler
    - Gradient clipping
    - Validation every epoch
    - Detailed logging every 10 epochs
    - Save best model based on validation loss
    """
    brain_dim = X_train.shape[1]
    clip_dim = Y_train.shape[1]
    
    print(f"\n{'='*60}")
    print("MODEL ARCHITECTURE")
    print(f"{'='*60}")
    print(f"Input dimension (brain voxels): {brain_dim}")
    print(f"Output dimension (CLIP):        {clip_dim}")
    print(f"Hidden dimension:               {HIDDEN_DIM}")
    print(f"Architecture:")
    print(f"  {brain_dim} → Linear({HIDDEN_DIM}) → LayerNorm → GELU → Dropout({DROPOUT_1})")
    print(f"  → ResidualBlock({HIDDEN_DIM}, dropout=0.2) × 2")
    print(f"  → Linear(2048) → LayerNorm → GELU → Dropout({DROPOUT_2})")
    print(f"  → Linear({clip_dim})")
    
    # Build model
    model = ImprovedBrainEncoder(brain_dim, clip_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params:,}")
    
    # Determine batch size based on dataset size
    n_samples = len(X_train)
    batch_size = BATCH_SIZE if n_samples >= 500 else 32
    print(f"\nBatch size: {batch_size}")
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).to(device),
        torch.from_numpy(Y_train).to(device)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val).to(device),
        torch.from_numpy(Y_val).to(device)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False)
    
    # Optimizer: AdamW (Adam with decoupled weight decay)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Scheduler: ReduceLROnPlateau (reduce LR when val loss plateaus)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        verbose=True
    )
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training log
    training_log = []
    
    print(f"\n{'='*60}")
    print(f"TRAINING ({EPOCHS} epochs)")
    print(f"{'='*60}")
    print(f"{'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>11} | "
          f"{'CosSim':>7} | {'Pearson':>7} | {'Top-1':>6} | {'Top-5':>6} | {'LR':>10}")
    print("-" * 90)
    
    for epoch in range(1, EPOCHS + 1):
        # ═══════════════════════════════════════════════════════════════════
        # TRAINING PHASE
        # ═══════════════════════════════════════════════════════════════════
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_cosine = 0.0
        epoch_contrast = 0.0
        n_batches = 0
        
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(x_batch)
            
            # L2 normalize predictions before computing loss
            pred_norm = F.normalize(pred, dim=-1, p=2)
            
            # Combined loss
            loss, l_mse, l_cos, l_con = combined_loss(pred_norm, y_batch, TEMPERATURE)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate losses
            epoch_loss += loss.item()
            epoch_mse += l_mse
            epoch_cosine += l_cos
            epoch_contrast += l_con
            n_batches += 1
        
        # Average training losses
        avg_train_loss = epoch_loss / n_batches
        
        # ═══════════════════════════════════════════════════════════════════
        # VALIDATION PHASE
        # ═══════════════════════════════════════════════════════════════════
        val_loss, val_cos_sim, val_pearson, val_top1, val_top5 = validate(
            model, val_loader, device
        )
        
        # Learning rate scheduling based on validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        training_log.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'train_mse': epoch_mse / n_batches,
            'train_cosine': epoch_cosine / n_batches,
            'train_contrast': epoch_contrast / n_batches,
            'val_loss': val_loss,
            'val_cos_sim': val_cos_sim,
            'val_pearson': val_pearson,
            'val_top1': val_top1,
            'val_top5': val_top5,
            'learning_rate': current_lr
        })
        
        # Print every 10 epochs or first/last epoch
        if epoch % 10 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"{epoch:>6} | {avg_train_loss:>11.4f} | {val_loss:>11.4f} | "
                  f"{val_cos_sim:>7.4f} | {val_pearson:>7.4f} | "
                  f"{val_top1:>6.2%} | {val_top5:>6.2%} | {current_lr:>10.2e}")
        
        # ═══════════════════════════════════════════════════════════════════
        # EARLY STOPPING & BEST MODEL SAVING
        # ═══════════════════════════════════════════════════════════════════
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model state
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"         ✅ New best model! Val loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⏹️  Early stopping triggered at epoch {epoch}")
                print(f"   Best validation loss: {best_val_loss:.4f}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        print(f"\n✅ Loaded best model from validation")
    
    return model, training_log


# ─────────────────────────────────────────────────────────────────────────────
# 8. TEST EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_test_set(model: ImprovedBrainEncoder, scaler: StandardScaler,
                     X_test_raw: np.ndarray, Y_test_raw: np.ndarray,
                     device: torch.device, name: str) -> np.ndarray:
    """
    Evaluate model on test set with comprehensive metrics.
    
    Returns L2-normalized predictions for Step 06 image generation.
    """
    model.eval()
    
    print(f"\n{'='*60}")
    print(f"EVALUATING: {name}")
    print(f"{'='*60}")
    print(f"Test samples: {len(X_test_raw)}")
    
    # Preprocess
    X_test = scaler.transform(X_test_raw).astype(np.float32)
    Y_test = Y_test_raw / (np.linalg.norm(Y_test_raw, axis=1, keepdims=True) + 1e-8)
    Y_test = Y_test.astype(np.float32)
    
    # Predict in batches
    X_tensor = torch.from_numpy(X_test).to(device)
    chunk_size = 128
    preds = []
    
    for i in range(0, len(X_tensor), chunk_size):
        x_chunk = X_tensor[i:i+chunk_size]
        pred_chunk = model(x_chunk)
        # L2 normalize predictions
        pred_norm = F.normalize(pred_chunk, dim=-1, p=2)
        preds.append(pred_norm.cpu().numpy())
    
    Y_pred = np.vstack(preds)
    
    # Compute metrics
    cos_sim = compute_cosine_similarity(Y_test, Y_pred)
    pearson = compute_pearson_r(Y_test, Y_pred)
    retrieval = compute_retrieval_accuracy(Y_test, Y_pred, k_values=[1, 5])
    
    print(f"  Cosine similarity:       {cos_sim:.4f}")
    print(f"  Pearson r:               {pearson:.4f}")
    print(f"  Top-1 retrieval:         {retrieval['top1']:.2%}")
    print(f"  Top-5 retrieval:         {retrieval['top5']:.2%}")
    
    # Check for collapse
    pred_norms = np.linalg.norm(Y_pred, axis=1)
    print(f"  Prediction norm mean:    {pred_norms.mean():.4f} (should be ~1.0)")
    print(f"  Prediction norm std:     {pred_norms.std():.4f} (should be ~0.0)")
    
    pred_std_per_dim = Y_pred.std(axis=0).mean()
    print(f"  Per-dimension std:       {pred_std_per_dim:.4f} (higher = less collapse)")
    
    if pred_std_per_dim < 0.01:
        print(f"  ⚠️  WARNING: Low variance - possible embedding collapse!")
    
    return Y_pred


# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"DEVICE & ENVIRONMENT")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    
    # ═════════════════════════════════════════════════════════════════════════
    # LOAD TRAINING DATA
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("LOADING TRAINING DATA")
    print(f"{'='*60}")
    
    try:
        X_raw = np.load(os.path.join(BRAIN_DIR, "train_mixed_brain.npy"))
        Y_raw = np.load(os.path.join(CLIP_DIR, "train_mixed_clip_embeds.npy"))
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Make sure you have run steps 02-04 first!")
        return
    
    # Sanity check and trim if needed
    X_raw, Y_raw = sanity_check_data(X_raw, Y_raw, "Training Data")
    
    # ═════════════════════════════════════════════════════════════════════════
    # TRAIN/VALIDATION SPLIT
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("TRAIN/VALIDATION SPLIT")
    print(f"{'='*60}")
    
    X_train_raw, X_val_raw, Y_train_raw, Y_val_raw = train_test_split(
        X_raw, Y_raw,
        test_size=VAL_SPLIT,
        random_state=RANDOM_STATE
    )
    
    print(f"Training samples:   {len(X_train_raw)} ({(1-VAL_SPLIT)*100:.0f}%)")
    print(f"Validation samples: {len(X_val_raw)} ({VAL_SPLIT*100:.0f}%)")
    
    # ═════════════════════════════════════════════════════════════════════════
    # PREPROCESS
    # ═════════════════════════════════════════════════════════════════════════
    X_train, Y_train, X_val, Y_val, scaler = preprocess_data(
        X_train_raw, Y_train_raw, X_val_raw, Y_val_raw
    )
    
    # ═════════════════════════════════════════════════════════════════════════
    # TRAIN
    # ═════════════════════════════════════════════════════════════════════════
    model, training_log = train_model(X_train, Y_train, X_val, Y_val, device)
    
    # ═════════════════════════════════════════════════════════════════════════
    # SAVE MODEL & LOGS
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("SAVING MODEL & ARTIFACTS")
    print(f"{'='*60}")
    
    model_path = os.path.join(MODELS_DIR, "csi1_mlp_encoder_best.pt")
    scaler_path = os.path.join(MODELS_DIR, "csi1_mlp_scaler.pkl")
    log_path = os.path.join(MODELS_DIR, "training_log.csv")
    
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    pd.DataFrame(training_log).to_csv(log_path, index=False)
    
    print(f"✅ Model saved:  {model_path}")
    print(f"✅ Scaler saved: {scaler_path}")
    print(f"✅ Log saved:    {log_path}")
    
    # ═════════════════════════════════════════════════════════════════════════
    # EVALUATE ON TEST SETS
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("TEST SET EVALUATION")
    print(f"{'='*60}")
    
    # Test A: Objects (ImageNet)
    try:
        X_obj = np.load(os.path.join(BRAIN_DIR, "test_objects_imagenet_brain.npy"))
        Y_obj = np.load(os.path.join(CLIP_DIR, "test_objects_imagenet_clip_embeds.npy"))
        
        # Trim if needed
        n_obj = min(len(X_obj), len(Y_obj))
        X_obj, Y_obj = X_obj[:n_obj], Y_obj[:n_obj]
        
        Y_pred_obj = evaluate_test_set(
            model, scaler, X_obj, Y_obj, device, "Objects (ImageNet)"
        )
        
        # Save predictions for Step 06
        pred_obj_path = os.path.join(MODELS_DIR, "pred_objects_mlp.npy")
        np.save(pred_obj_path, Y_pred_obj)
        print(f"✅ Saved: {pred_obj_path}")
        
    except FileNotFoundError:
        print("\n⚠️  Object test files not found - skipping")
        Y_pred_obj = None
    
    # Test B: Scenes (COCO)
    try:
        X_scene = np.load(os.path.join(BRAIN_DIR, "test_scenes_coco_brain.npy"))
        Y_scene = np.load(os.path.join(CLIP_DIR, "test_scenes_coco_clip_embeds.npy"))
        
        # Trim if needed
        n_scene = min(len(X_scene), len(Y_scene))
        X_scene, Y_scene = X_scene[:n_scene], Y_scene[:n_scene]
        
        Y_pred_scene = evaluate_test_set(
            model, scaler, X_scene, Y_scene, device, "Scenes (COCO)"
        )
        
        # Save predictions for Step 06
        pred_scene_path = os.path.join(MODELS_DIR, "pred_scenes_mlp.npy")
        np.save(pred_scene_path, Y_pred_scene)
        print(f"✅ Saved: {pred_scene_path}")
        
    except FileNotFoundError:
        print("\n⚠️  Scene test files not found - skipping")
        Y_pred_scene = None
    
    # ═════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print("\n📊 What to check:")
    print("  1. Validation metrics should show:")
    print("     - Cosine similarity > 0.5 (good), > 0.7 (excellent)")
    print("     - Top-1 retrieval > 20% (working), > 50% (strong)")
    print("     - Top-5 retrieval > 50% (working), > 80% (strong)")
    print("\n  2. Test set prediction norms should be ~1.0")
    print("     (indicates proper L2 normalization)")
    print("\n  3. Per-dimension std should be > 0.05")
    print("     (indicates predictions are diverse, not collapsed)")
    print("\n  4. Training log saved to training_log.csv")
    print("     - Plot val_loss, val_cos_sim, val_top1 over epochs")
    print("     - Check if early stopping was triggered appropriately")
    print("\n📁 Next step:")
    print("  Run Step 06 (image generation) with:")
    print("    - pred_objects_mlp.npy")
    print("    - pred_scenes_mlp.npy")
    print("\n  If images still look noisy:")
    print("    - Check retrieval accuracy (should be >30% top-1)")
    print("    - Increase WEIGHT_CONTRAST to 2.0 or 3.0")
    print("    - Decrease TEMPERATURE to 0.03 (harder negatives)")
    print("    - Train longer (increase EPOCHS to 300)")


if __name__ == "__main__":
    main()