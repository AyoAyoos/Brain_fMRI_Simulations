# Brain fMRI Simulations (BOLD5000) — Data Prep + Brain→Image Baseline

This repository contains a **step-by-step pipeline** to:

- build **train/test splits** from BOLD5000 stimulus images (ImageNet vs COCO vs SUN/Scene)
- merge **BOLD5000 ROI fMRI betas** for a subject (currently `CSI1`)
- extract **CLIP image embeddings** for the same images
- **align** (average repeated trials) brain data to the split lists
- train a **brain → CLIP** translator (Ridge baseline or MLP variant)
- optionally generate images with **Stable Diffusion 1.5** or **Stable Diffusion XL (SDXL)**
- compute **pixel metrics** (MSE / SSIM) between real and reconstructed images

> Important: the scripts currently use **absolute Windows paths** (e.g. `D:/PBL_6/DATA_prep/...`).  
> If your folder is in a different location, edit the configuration blocks at the top of each script.

---

## Folder structure (expected)

At a high level, your workspace is expected to look like:

```
DATA_prep/
  scripts/
    01_data_splitter.py
    02_brain_feature_merger.py
    03_clip_feature_extractor.py
    04_data_aligner.py
    05_train_translator.py
    05b_train_mlp_translator.py
    06_generate_images.py
    06b_generate_images.py
    06c_generate_images_sdxl.py
    07_pixel_metrics.py

  Original_Images/
    ImageNet/
    COCO/
    Scene/   (or SUN/)

  CSI1/
    CSI1_sess01/
      run01.txt ...
    ...

  BOLD5000_GLMsingle_ROI_betas/
    BOLD5000_GLMsingle_ROI_betas/
      py/
        CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_allses_LHEarlyVis.npy
        ...

  output/
    splits/
    brain_features/
    clip_features/
    aligned_brain/
    models/
    reconstructions/
```

---

## Setup

### 1) Create and activate a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### 2) Install dependencies

You can install packages as-needed while running the steps. Typical requirements include:

- `numpy`, `pandas`, `scikit-learn`, `scipy`
- `Pillow`
- `torch`
- `transformers`
- `diffusers`
- `scikit-image`

Example:

```bash
pip install numpy pandas scikit-learn scipy pillow torch transformers diffusers scikit-image joblib
```

---

## Pipeline (run order)

### Step 1 — Create splits (ImageNet objects vs COCO scenes + training mix)

Script: `scripts/01_data_splitter.py`

- Reads images from:
  - `Original_Images/ImageNet/`
  - `Original_Images/COCO/`
  - `Original_Images/Scene/` (falls back to `SUN/` if not found)
- Writes CSVs to `output/splits/`:
  - `train_mixed.csv`
  - `test_objects_imagenet.csv` (100 random objects)
  - `test_scenes_coco.csv` (100 random scenes)

Run:

```bash
python scripts/01_data_splitter.py
```

### Step 2 — Merge ROI betas for a subject into one matrix

Script: `scripts/02_brain_feature_merger.py`

- Loads multiple ROI `.npy` beta matrices for `CSI1`
- Horizontally concatenates them into one matrix of shape:
  - `(num_trials, num_voxels_total)`
- Saves:
  - `output/brain_features/CSI1_merged_brain.npy`

Run:

```bash
python scripts/02_brain_feature_merger.py
```

### Step 3 — Extract CLIP embeddings for each split

Script: `scripts/03_clip_feature_extractor.py`

- Uses `openai/clip-vit-large-patch14`
- For each CSV in `output/splits/`, generates an embedding matrix and saves it to:
  - `output/clip_features/*_clip_embeds.npy`

Run:

```bash
python scripts/03_clip_feature_extractor.py
```

### Step 4 — Align brain trials to image filenames (and average repeats)

Script: `scripts/04_data_aligner.py`

- Parses BOLD5000 presentation lists in `CSI1/CSI1_sessXX/runYY.txt`
- Maps image filename → trial indices in the merged brain matrix
- For each split, averages repeated trials, producing:
  - `output/aligned_brain/train_mixed_brain.npy`
  - `output/aligned_brain/test_objects_imagenet_brain.npy`
  - `output/aligned_brain/test_scenes_coco_brain.npy`

Run:

```bash
python scripts/04_data_aligner.py
```

### Step 5A — Train brain→CLIP translator (Ridge regression baseline)

Script: `scripts/05_train_translator.py`

- Trains a `StandardScaler + RidgeCV` model to predict CLIP vectors from brain features
- Saves:
  - `output/models/csi1_brain_translator.pkl`
  - predictions for the two test sets:
    - `output/models/pred_objects.npy`
    - `output/models/pred_scenes.npy`

Run:

```bash
python scripts/05_train_translator.py
```

### Step 5B — Train brain→CLIP translator (MLP + contrastive loss)

Script: `scripts/05b_train_mlp_translator.py`

- Trains a nonlinear `BrainEncoder` MLP with:
  - MSE loss
  - contrastive loss term
- Saves:
  - `output/models/csi1_mlp_encoder.pt`
  - `output/models/csi1_mlp_scaler.pkl`
  - prediction files:
    - `output/models/pred_objects_mlp.npy`
    - `output/models/pred_scenes_mlp.npy`

Run:

```bash
python scripts/05b_train_mlp_translator.py
```

### Step 6A (optional) — Generate images (Stable Diffusion 1.5)

Script: `scripts/06_generate_images.py` (or `scripts/06b_generate_images.py`)

- Loads prediction files from Step 5 (Ridge or MLP outputs)
- Injects projected embeddings as `prompt_embeds` into SD 1.5 (`runwayml/stable-diffusion-v1-5`)
- Supports direct generation and retrieval+img2img style reconstructions
- Saves PNG reconstructions under `output/reconstructions/<run_label>/...`

Run:

```bash
python scripts/06_generate_images.py
```

Notes:
- This step is compute-heavy and may require a GPU.
- Generated file names can include strategy prefixes such as `recon_direct_XXX.png` or `recon_retrieval_XXX.png`.

### Step 6B (optional) — Generate images with SDXL (recommended quality)

Script: `scripts/06c_generate_images_sdxl.py`

- Uses SDXL base model: `stabilityai/stable-diffusion-xl-base-1.0`
- Projects brain predictions to SDXL conditioning:
  - sequence embeddings: `(1, 77, 2048)`
  - pooled embeddings: `(1, 1280)`
- Produces 1024x1024 images and supports:
  - Strategy A: direct SDXL generation
  - Strategy B: nearest-neighbor retrieval + SDXL img2img

Run:

```bash
python scripts/06c_generate_images_sdxl.py
```

Notes:
- First SDXL run downloads a large model (~6.5GB).
- For CUDA, fp16 and CPU offload are used in-script for lower VRAM usage.

### Step 7 (optional) — Pixel-level metrics (MSE / SSIM)

Script: `scripts/07_pixel_metrics.py`

- Compares reconstructed images with the ground-truth images
- Uses `test_scenes_coco.csv` (configured in the script) to map index → filename/folder
- Prints:
  - Average MSE
  - Average SSIM

Run:

```bash
python scripts/07_pixel_metrics.py
```

Notes:
- Set `RECON_DIR` / folder selection to the specific reconstruction output you want to score.
- If evaluating SDXL outputs, use 1024x1024 resizing for fair comparison.

---

## Repro tips

- **Paths**: Update the configuration section at the top of each script to match your local folder.
- **Keep GitHub light**: Don’t push datasets (`Original_Images/`, ROI `.npy`), `output/` artifacts, or `venv/`.
- **Share data via links**: Put download links for BOLD5000/ROI files in this README instead of committing them.

---

## License

Add a license if you plan to share publicly (MIT is common for code).

