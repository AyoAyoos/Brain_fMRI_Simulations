"""
05b_train_mlp_translator.py
============================
Drop-in replacement for 05_train_translator.py.

WHAT CHANGED VS RIDGE:
  - Ridge is a single linear layer:  y = Wx + b
  - This MLP is:  brain → 2048 → 1024 → 768(CLIP)
    with LayerNorm, GELU activations, and Dropout at each layer.
  - Loss = MSE  +  λ * contrastive  (both in one backward pass)
  - We save a .pt checkpoint so you can reload and probe it later.
  - Pearson-r evaluation is kept identical to your Ridge script so
    results are directly comparable.

HOW TO RUN:
  python 05b_train_mlp_translator.py

REQUIREMENTS (add to your env if missing):
  pip install torch numpy scikit-learn scipy joblib
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import joblib

# ─────────────────────────────────────────
# CONFIGURATION  — same paths as your existing scripts
# ─────────────────────────────────────────
BRAIN_DIR  = r"D:/PBL_6/DATA_prep/output/aligned_brain"
CLIP_DIR   = r"D:/PBL_6/DATA_prep/output/clip_features"
MODELS_DIR = r"D:/PBL_6/DATA_prep/output/models"

# Training hyper-parameters
# ── start here, tune if val-loss plateaus or spikes ──
EPOCHS        = 80          # more than Ridge needs, but still fast
BATCH_SIZE    = 32          # keep small — fMRI datasets are tiny (~1000 samples)
LEARNING_RATE = 3e-4        # Adam default; lower to 1e-4 if loss is noisy
WEIGHT_DECAY  = 1e-4        # L2 regularisation on weights
DROPOUT_RATE  = 0.3         # higher = more regularisation; try 0.2–0.5
LAMBDA_CON    = 0.1         # weight for contrastive loss term; 0 = pure MSE
TEMPERATURE   = 0.07        # contrastive softmax temperature (CLIP default)
HIDDEN_DIMS   = [2048, 1024] # MLP hidden layer sizes; add/remove layers here
CLIP_DIM      = 1024          # output dim — matches openai/clip-vit-large-patch14
# ─────────────────────────────────────────


# ─────────────────────────────────────────
# 1.  MODEL DEFINITION
# ─────────────────────────────────────────
class BrainEncoder(nn.Module):
    """
    Maps a fMRI voxel vector → CLIP embedding space.

    Architecture:
        input (voxels)
            │
        Linear → LayerNorm → GELU → Dropout
            │
        Linear → LayerNorm → GELU → Dropout
            │
        Linear (no activation — raw CLIP space)

    Why LayerNorm instead of BatchNorm?
        BatchNorm shifts with batch statistics — bad for tiny
        neuroimaging batches.  LayerNorm normalises per-sample,
        which is stabler here.

    Why GELU instead of ReLU?
        GELU is smooth near 0 and is what most modern vision/NLP
        models use. In practice it slightly outperforms ReLU on
        representation learning tasks.
    """

    def __init__(self, brain_dim: int, hidden_dims: list, clip_dim: int,
                 dropout: float = 0.3):
        super().__init__()

        layers = []
        in_dim = brain_dim

        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim

        # Final projection — no activation so output lives in CLIP space
        layers.append(nn.Linear(in_dim, clip_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────
# 2.  LOSS FUNCTIONS
# ─────────────────────────────────────────
def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Standard pixel-wise loss — forces predicted vector close to true CLIP."""
    return F.mse_loss(pred, target)


def contrastive_loss(pred: torch.Tensor, target: torch.Tensor,
                     temperature: float = 0.07) -> torch.Tensor:
    """
    Contrastive (CLIP-style) loss.

    For a batch of N (brain, image) pairs:
      - brain[i] should be closest to clip[i]  (positive pair)
      - brain[i] should be far from clip[j≠i] (negative pairs)

    This forces the model to distinguish *which image* a brain signal
    corresponds to, not just reconstruct an average CLIP vector.

    If batch_size == 1 this is undefined — we skip it gracefully.
    """
    if pred.shape[0] < 2:
        return torch.tensor(0.0, device=pred.device)

    # L2-normalise so dot-product = cosine similarity
    pred_norm   = F.normalize(pred,   dim=-1)
    target_norm = F.normalize(target, dim=-1)

    # (N, N) similarity matrix
    logits = torch.matmul(pred_norm, target_norm.T) / temperature

    # Diagonal entries are the correct pairs
    labels = torch.arange(pred.shape[0], device=pred.device)

    # Symmetric cross-entropy: brain→image AND image→brain
    loss = (F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.T, labels)) / 2.0
    return loss


def combined_loss(pred: torch.Tensor, target: torch.Tensor,
                  lambda_con: float = 0.1, temperature: float = 0.07):
    """
    Total loss = MSE  +  λ * contrastive

    Intuition:
      MSE alone makes predictions 'close' in L2 distance.
      Contrastive additionally makes each prediction uniquely
      match its own image rather than the average image.
    """
    l_mse = mse_loss(pred, target)
    l_con = contrastive_loss(pred, target, temperature)
    return l_mse + lambda_con * l_con, l_mse.item(), l_con.item()


# ─────────────────────────────────────────
# 3.  EVALUATION  (same metric as Ridge script)
# ─────────────────────────────────────────
def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                          label: str) -> float:
    """
    Per-image Pearson-r between true and predicted CLIP vectors,
    then averaged.  Same function as in 05_train_translator.py so
    scores are directly comparable.
    """
    correlations = []
    for i in range(len(y_true)):
        corr, _ = pearsonr(y_true[i], y_pred[i])
        correlations.append(corr)
    avg = float(np.mean(correlations))
    print(f"   📊 {label}: r = {avg:.4f}  (max = {np.max(correlations):.4f})")
    return avg


# ─────────────────────────────────────────
# 4.  TRAINING LOOP
# ─────────────────────────────────────────
def train_model(X_train_raw: np.ndarray, Y_train: np.ndarray,
                device: torch.device) -> tuple:

    CLIP_DIM = Y_train.shape[1]
    print(f"   Auto-detected CLIP_DIM: {CLIP_DIM}")
    """
    Fits the StandardScaler, builds the MLP, runs the training loop.
    Returns (model, scaler) so both can be saved and reused at test time.
    """

    # --- 4a. Normalise brain data (same as Ridge pipeline) ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    Y_train  = Y_train.astype(np.float32)

    # --- 4b. Tensors ---
    X_t = torch.from_numpy(X_train).to(device)
    Y_t = torch.from_numpy(Y_train).to(device)
    dataset = TensorDataset(X_t, Y_t)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         drop_last=True)  # drop_last avoids batch-size-1 edge case

    # --- 4c. Build model ---
    brain_dim = X_train.shape[1]
    model = BrainEncoder(brain_dim, HIDDEN_DIMS, CLIP_DIM, DROPOUT_RATE).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Model parameters: {total_params:,}")
    print(f"   Input voxels:     {brain_dim}")
    print(f"   Architecture:     {brain_dim} → {' → '.join(map(str, HIDDEN_DIMS))} → {CLIP_DIM}")

    # --- 4d. Optimiser ---
    # AdamW is Adam + proper weight decay (decoupled).
    # Weight decay acts as L2 regularisation — critical for noisy fMRI data.
    optimiser = torch.optim.AdamW(model.parameters(),
                                   lr=LEARNING_RATE,
                                   weight_decay=WEIGHT_DECAY)

    # Cosine annealing: learning rate starts at LR and smoothly decays to 0.
    # Better than constant LR — avoids overshooting at the end.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=EPOCHS)

    # --- 4e. Training loop ---
    print(f"\n   Training for {EPOCHS} epochs...")
    print(f"   {'Epoch':>6}  {'Total':>10}  {'MSE':>10}  {'Contrastive':>12}")
    print("   " + "─" * 44)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        epoch_total = 0.0
        epoch_mse   = 0.0
        epoch_con   = 0.0
        n_batches   = 0

        for x_batch, y_batch in loader:
            optimiser.zero_grad()
            pred = model(x_batch)
            loss, l_mse, l_con = combined_loss(
                pred, y_batch, LAMBDA_CON, TEMPERATURE)
            loss.backward()

            # Gradient clipping: prevents exploding gradients
            # (common with noisy fMRI signals)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimiser.step()
            epoch_total += loss.item()
            epoch_mse   += l_mse
            epoch_con   += l_con
            n_batches   += 1

        scheduler.step()

        # Print every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            avg_total = epoch_total / n_batches
            avg_mse   = epoch_mse   / n_batches
            avg_con   = epoch_con   / n_batches
            print(f"   {epoch:>6}  {avg_total:>10.4f}  {avg_mse:>10.4f}  {avg_con:>12.4f}")

    return model, scaler


# ─────────────────────────────────────────
# 5.  INFERENCE HELPER
# ─────────────────────────────────────────
@torch.no_grad()
def predict(model: BrainEncoder, scaler: StandardScaler,
            X_raw: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Scales raw brain data with the FITTED scaler, runs through model,
    returns numpy predictions.
    """
    model.eval()
    X_scaled = scaler.transform(X_raw).astype(np.float32)
    X_t      = torch.from_numpy(X_scaled).to(device)

    # Run in chunks to avoid OOM on large test sets
    chunk_size = 128
    preds = []
    for i in range(0, len(X_t), chunk_size):
        preds.append(model(X_t[i : i + chunk_size]).cpu().numpy())

    return np.vstack(preds)


# ─────────────────────────────────────────
# 6.  MAIN
# ─────────────────────────────────────────
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # ── LOAD TRAINING DATA ──────────────────────────────────────────────
    print("\n⏳ Loading training data...")
    try:
        X_train_raw = np.load(os.path.join(BRAIN_DIR, "train_mixed_brain.npy"))
        Y_train     = np.load(os.path.join(CLIP_DIR,  "train_mixed_clip_embeds.npy"))
    except FileNotFoundError as e:
        print(f"❌ {e}\n   Have you run scripts 02–04 first?")
        return

    # Trim to same length (alignment can drop a few rows)
    n = min(len(X_train_raw), len(Y_train))
    X_train_raw, Y_train = X_train_raw[:n], Y_train[:n]
    print(f"   Training samples: {n}")
    print(f"   Brain voxels:     {X_train_raw.shape[1]}")
    print(f"   CLIP dim:         {Y_train.shape[1]}")

    # ── TRAIN ──────────────────────────────────────────────────────────
    print("\n🧠 Training MLP BrainEncoder...")
    model, scaler = train_model(X_train_raw, Y_train, device)

    # ── SAVE ───────────────────────────────────────────────────────────
    model_path  = os.path.join(MODELS_DIR, "csi1_mlp_encoder.pt")
    scaler_path = os.path.join(MODELS_DIR, "csi1_mlp_scaler.pkl")
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\n✅ Saved model  → {model_path}")
    print(f"✅ Saved scaler → {scaler_path}")

    # ── TEST A: OBJECTS ────────────────────────────────────────────────
    print("\n🔎 Evaluating on test sets...")
    score_obj = score_scene = None

    try:
        X_obj = np.load(os.path.join(BRAIN_DIR, "test_objects_imagenet_brain.npy"))
        Y_obj = np.load(os.path.join(CLIP_DIR,  "test_objects_imagenet_clip_embeds.npy"))
        n_obj = min(len(X_obj), len(Y_obj))
        X_obj, Y_obj = X_obj[:n_obj], Y_obj[:n_obj]

        Y_pred_obj = predict(model, scaler, X_obj, device)
        score_obj  = evaluate_predictions(Y_obj, Y_pred_obj, "Objects (ImageNet)")

        # Save predictions for Step 6 (image generation)
        np.save(os.path.join(MODELS_DIR, "pred_objects_mlp.npy"), Y_pred_obj)

    except FileNotFoundError:
        print("   ⚠️  Object test files not found — skipping.")

    # ── TEST B: SCENES ─────────────────────────────────────────────────
    try:
        X_scene = np.load(os.path.join(BRAIN_DIR, "test_scenes_coco_brain.npy"))
        Y_scene = np.load(os.path.join(CLIP_DIR,  "test_scenes_coco_clip_embeds.npy"))
        n_sc = min(len(X_scene), len(Y_scene))
        X_scene, Y_scene = X_scene[:n_sc], Y_scene[:n_sc]

        Y_pred_scene = predict(model, scaler, X_scene, device)
        score_scene  = evaluate_predictions(Y_scene, Y_pred_scene, "Scenes (COCO)")

        np.save(os.path.join(MODELS_DIR, "pred_scenes_mlp.npy"), Y_pred_scene)

    except FileNotFoundError:
        print("   ⚠️  Scene test files not found — skipping.")

    # ── FINAL VERDICT ──────────────────────────────────────────────────
    print("\n📝 RESULTS SUMMARY")
    print("   ─────────────────────────────────────────")
    if score_obj is not None and score_scene is not None:
        diff = score_obj - score_scene
        print(f"   Objects r  : {score_obj:.4f}")
        print(f"   Scenes  r  : {score_scene:.4f}")
        print(f"   Gap        : {diff:+.4f}")
        if diff > 0:
            print("   ✅ Objects decoded better — hypothesis holds with MLP too.")
        else:
            print("   ❓ Scenes equal/better — interesting deviation from Ridge result.")
        print("\n   Compare these scores against your Ridge baseline (05_train_translator.py)")
        print("   to quantify how much the nonlinear model actually helps.")

    print("\n   Prediction files saved:")
    print(f"   → pred_objects_mlp.npy  (use in 06_generate_images.py)")
    print(f"   → pred_scenes_mlp.npy")


# ─────────────────────────────────────────
# HOW TO RELOAD THE MODEL LATER
# (copy this into any script that needs predictions)
# ─────────────────────────────────────────
def load_trained_model(models_dir: str, brain_dim: int,
                       device: torch.device):
    """
    Example of how to reload the saved model after training.

    Usage:
        model, scaler = load_trained_model(MODELS_DIR, brain_dim, device)
        preds = predict(model, scaler, X_new, device)
    """
    scaler = joblib.load(os.path.join(models_dir, "csi1_mlp_scaler.pkl"))
    model  = BrainEncoder(brain_dim, HIDDEN_DIMS, CLIP_DIM, DROPOUT_RATE)
    model.load_state_dict(
        torch.load(os.path.join(models_dir, "csi1_mlp_encoder.pt"),
                   map_location=device))
    model = model.to(device).eval()
    return model, scaler


if __name__ == "__main__":
    main()