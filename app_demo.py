import os
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import streamlit.components.v1 as components
from scipy.stats import gamma

# For Metrics
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error

# =====================================================================
# CRITICAL FIX: Set Matplotlib to headless BEFORE importing nilearn
# =====================================================================
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# For Brain Viz
import nilearn
from nilearn import datasets, surface, plotting

# =====================================================================
# CONFIGURATION & STATE
# =====================================================================
st.set_page_config(page_title="fMRI Visual Decoder", layout="wide", page_icon="🧠")

# Default paths matching the local setup
DEFAULT_IMAGES_DIR = r"D:\PBL_6\DATA_prep\Original_Images"
DEFAULT_RECON_DIR = r"D:\PBL_6\DATA_prep\output\reconstructions"
DEFAULT_MODELS_DIR = r"D:\PBL_6\DATA_prep\output\models"
DEFAULT_SPLITS_DIR = r"D:/PBL_6/DATA_prep/output/splits"
SCRIPTS_DIR = "scripts"

# Use session state to persist configuration paths
if "paths" not in st.session_state:
    st.session_state.paths = {
        "images": DEFAULT_IMAGES_DIR,
        "recon": DEFAULT_RECON_DIR,
        "splits": DEFAULT_SPLITS_DIR,
        "models": DEFAULT_MODELS_DIR
    }

# Helper function to run scripts
def run_script(script_name):
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    if not os.path.exists(script_path):
        st.error(f"❌ Script not found: {script_path}")
        return
    
    with st.spinner(f"Running {script_name}... Check your terminal for details."):
        try:
            result = subprocess.run(["python", script_path], capture_output=True, text=True)
            if result.returncode == 0:
                st.success(f"✅ Successfully ran {script_name}")
                with st.expander("Show Output Logs"):
                    st.code(result.stdout)
            else:
                st.error(f"⚠️ Error running {script_name}")
                with st.expander("Show Error Logs"):
                    st.code(result.stderr)
        except Exception as e:
            st.error(f"Failed to execute: {str(e)}")

def load_image(path):
    try:
        return np.array(Image.open(path).convert("RGB").resize((512, 512)))
    except Exception as e:
        return None

# =====================================================================
# UI LAYOUT
# =====================================================================
st.title("🧠 Brain fMRI to Image Decoder")
st.markdown("A unified dashboard to execute the end-to-end MLP + SDXL pipeline, visualize brain ROIs, and evaluate reconstructed stimuli.")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    with st.expander("Edit Local Paths", expanded=False):
        st.session_state.paths["images"] = st.text_input("Original Images Dir", st.session_state.paths["images"])
        st.session_state.paths["recon"] = st.text_input("Reconstructions Dir", st.session_state.paths["recon"])
        st.session_state.paths["splits"] = st.text_input("Splits CSV Dir", st.session_state.paths["splits"])
        st.session_state.paths["models"] = st.text_input("Models / Preds Dir", st.session_state.paths["models"])

# Tabs - Added the Simulation Tab!
tab_pipeline, tab_brain, tab_simulation, tab_results = st.tabs([
    "🚀 Pipeline Control", 
    "🌐 3D Brain Explorer", 
    "📈 Neuroscience Simulation",
    "📊 Reconstruction Gallery"
])

# ---------------------------------------------------------------------
# TAB 1: PIPELINE CONTROL
# ---------------------------------------------------------------------
with tab_pipeline:
    st.header("Execute Full Pipeline Scripts")
    st.markdown("Follow the sequential workflow from data preparation to image generation.")
    
    # --- NEW: Workflow Status Panel ---
    with st.expander("📋 Workflow Status (File Tracker)", expanded=True):
        # Derive the base output directory from the splits path (e.g., 'D:/.../output')
        base_output_dir = os.path.dirname(st.session_state.paths["splits"])
        
        check_items = {
            "Splits: train_mixed.csv": os.path.join(st.session_state.paths["splits"], "train_mixed.csv"),
            "Splits: test_scenes_coco.csv": os.path.join(st.session_state.paths["splits"], "test_scenes_coco.csv"),
            "Splits: test_objects_imagenet.csv": os.path.join(st.session_state.paths["splits"], "test_objects_imagenet.csv"),
            "Brain: CSI1_merged_brain.npy": os.path.join(base_output_dir, "brain_features", "CSI1_merged_brain.npy"),
            "CLIP: train_mixed_clip_embeds.npy": os.path.join(base_output_dir, "clip_features", "train_mixed_clip_embeds.npy"),
            "Aligned: train_mixed_brain.npy": os.path.join(base_output_dir, "aligned_brain", "train_mixed_brain.npy"),
            "Models: csi1_mlp_encoder.pt": os.path.join(st.session_state.paths["models"], "csi1_mlp_encoder.pt"),
            "Models: pred_scenes_mlp.npy": os.path.join(st.session_state.paths["models"], "pred_scenes_mlp.npy"),
            "Models: pred_objects_mlp.npy": os.path.join(st.session_state.paths["models"], "pred_objects_mlp.npy"),
            "Recon: pred_scenes_mlp_sdxl": os.path.join(st.session_state.paths["recon"], "pred_scenes_mlp_sdxl"),
            "Recon: pred_objects_mlp_sdxl": os.path.join(st.session_state.paths["recon"], "pred_objects_mlp_sdxl"),
        }
        
        status_cols = st.columns(3)
        for i, (label, file_path) in enumerate(check_items.items()):
            exists = os.path.exists(file_path)
            icon = "✅" if exists else "❌"
            status_cols[i % 3].caption(f"{icon} {label}")
            
    st.divider()
    
    st.subheader("Phase 1: Data Preparation")
    st.caption("Prepare splits, merge ROIs, extract image features, and align data.")
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    with col_d1:
        if st.button("▶ 01. Split Data", use_container_width=True, help="01_data_splitter.py"): run_script("01_data_splitter.py")
    with col_d2:
        if st.button("▶ 02. Merge Brain", use_container_width=True, help="02_brain_feature_merger.py"): run_script("02_brain_feature_merger.py")
    with col_d3:
        if st.button("▶ 03. Extract CLIP", use_container_width=True, help="03_clip_feature_extractor.py"): run_script("03_clip_feature_extractor.py")
    with col_d4:
        if st.button("▶ 04. Align Data", use_container_width=True, help="04_data_aligner.py"): run_script("04_data_aligner.py")
        
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Phase 2: Train Translator (Step 05)")
        st.info("Primary: Train an MLP network to map fMRI signals to CLIP embeddings (`05b_train_mlp_translator.py`).")
        if st.button("▶ Run MLP Training (Main Workflow)", use_container_width=True):
            run_script("05b_train_mlp_translator.py")
            
        with st.expander("Baseline Method (Optional)"):
            st.caption("Runs `05_train_translator.py` using simple Ridge Regression.")
            if st.button("▶ Run Ridge Training", use_container_width=True):
                run_script("05_train_translator.py")

    with col2:
        st.subheader("Phase 3: Generate Images (Step 06)")
        st.info("Primary: Pass predicted embeddings into SDXL/SSD-1B for high-res generation (`06c_generate_images_sdxl.py`).")
        if st.button("▶ Run SDXL Generation (Main Workflow)", use_container_width=True):
            run_script("06c_generate_images_sdxl.py")
            
        with st.expander("Baseline Method (Optional)"):
            st.caption("Runs `06_generate_images.py` using standard SD 1.5.")
            if st.button("▶ Run Image Generation (SD 1.5)", use_container_width=True):
                run_script("06_generate_images.py")

    st.divider()
    st.subheader("Phase 4: Evaluation (Step 07)")
    st.markdown("Pixel metrics calculation (`07_pixel_metrics.py`) runs dynamically in the **📊 Reconstruction Gallery** tab when you explore the generated results.")

# ---------------------------------------------------------------------
# TAB 2: 3D BRAIN EXPLORER
# ---------------------------------------------------------------------
with tab_brain:
    st.header("Interactive Cortical Surface")
    st.markdown("Visualize the standard brain mesh to understand ROI placements.")
    
    if st.button("Load 3D Brain Surface"):
        with st.spinner("Fetching standard fsaverage surface from Nilearn (this may take a moment on first run)..."):
            try:
                fsaverage = datasets.fetch_surf_fsaverage()
                # Create interactive view
                view = plotting.view_surf(
                    surf_mesh=fsaverage.infl_right, 
                    surf_map=fsaverage.sulc_right, 
                    bg_map=fsaverage.sulc_right,
                    cmap='Greys',
                    title="Right Hemisphere (Inflated)"
                )
                
                # Render HTML directly in Streamlit
                components.html(view.get_iframe(), height=600, width=800)
            except Exception as e:
                st.error(f"Could not load Brain Visualizer: {e}")

# ---------------------------------------------------------------------
# TAB 3: NEUROSCIENCE SIMULATION (New)
# ---------------------------------------------------------------------
with tab_simulation:
    st.header("Hemodynamic Response Function (HRF) Simulation")
    st.markdown("Based on `advanced_simulation.py`. Simulates how blood oxygen level dependent (BOLD) signals react to visual stimuli over time using a double-gamma model.")
    
    sim_col1, sim_col2 = st.columns([1, 2])
    
    with sim_col1:
        st.subheader("HRF Parameters")
        peak_time = st.slider("Peak Time", 2.0, 10.0, 6.0, help="Time to peak of the response")
        undershoot_time = st.slider("Undershoot Time", 10.0, 25.0, 16.0, help="Time to the post-stimulus undershoot")
        ratio = st.slider("Peak/Undershoot Ratio", 1.0, 10.0, 6.0)

    with sim_col2:
        # Generate time array
        t = np.linspace(0, 30, 100)
        
        # Calculate Double-Gamma using scipy
        peak_gamma = gamma.pdf(t, a=peak_time)
        undershoot_gamma = gamma.pdf(t, a=undershoot_time)
        hrf = peak_gamma - (undershoot_gamma / ratio)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t, hrf, color='#ff4b4b', linewidth=3)
        ax.set_title("Simulated BOLD Signal")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Signal Amplitude")
        ax.axhline(0, color='black', linewidth=0.8, linestyle="--")
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Render plot in Streamlit
        st.pyplot(fig)

# ---------------------------------------------------------------------
# TAB 4: RECONSTRUCTION GALLERY
# ---------------------------------------------------------------------
with tab_results:
    st.header("Visual Reconstruction & Metrics")
    
    # 1. Generation Version, Dataset & Strategy Selection
    col_sel1, col_sel2, col_sel3 = st.columns(3)
    
    with col_sel1:
        gen_version = st.selectbox(
            "Select Generation Version", 
            ["06", "06b", "06c"],
            index=2 # Default to 06c (SDXL)
        )
        
    with col_sel2:
        dataset_choice = st.selectbox(
            "Select Dataset Split", 
            ["Scenes (COCO)", "Objects (ImageNet)"]
        )
        
    with col_sel3:
        # Disable strategy selection if basic '06' version is chosen
        strategy_choice = st.selectbox(
            "Select Strategy", 
            ["direct", "retrieval"],
            disabled=(gen_version == "06")
        )
    
    is_scenes = "Scenes" in dataset_choice
    csv_file = "test_scenes_coco.csv" if is_scenes else "test_objects_imagenet.csv"
    csv_path = os.path.join(st.session_state.paths["splits"], csv_file)
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.success(f"✅ Loaded {len(df)} entries from {csv_file}")
        
        # 2. Image Selector
        idx = st.slider("Select Trial Index", 0, max(0, len(df)-1), 0)
        
        if len(df) > 0:
            row = df.iloc[idx]
            
            # Ground Truth Image Path
            real_img_path = os.path.join(st.session_state.paths["images"], str(row.get("folder", "")), str(row.get("filename", "")))
            
            # 3. Dynamic Reconstruction Image Path Logic
            if gen_version == "06":
                # Rule 1: Basic Generation Output
                run_label = "pred_scenes" if is_scenes else "pred_objects"
                recon_img_path = os.path.join(st.session_state.paths["models"], run_label, f"recon_{idx:03d}.png")
                
                # Graceful fallback in case they saved basic recons in the recon folder instead of models
                if not os.path.exists(recon_img_path):
                    fallback_path = os.path.join(st.session_state.paths["recon"], run_label, f"recon_{idx:03d}.png")
                    if os.path.exists(fallback_path): recon_img_path = fallback_path

            elif gen_version == "06b":
                # Rule 2: Advanced Generation Output (SD 1.5)
                run_label = "pred_scenes" if is_scenes else "pred_objects"
                recon_img_path = os.path.join(st.session_state.paths["recon"], run_label, strategy_choice, f"recon_{strategy_choice}_{idx:03d}.png")

            else: # "06c"
                # Rule 3: SDXL / SSD-1B Generation Output
                run_label = "pred_scenes_mlp_sdxl" if is_scenes else "pred_objects_mlp_sdxl"
                recon_img_path = os.path.join(st.session_state.paths["recon"], run_label, strategy_choice, f"recon_{strategy_choice}_{idx:03d}.png")
            
            # 4. Load & Display Images
            col_img1, col_img2 = st.columns(2)
            
            real_img = load_image(real_img_path)
            recon_img = load_image(recon_img_path)
            
            with col_img1:
                st.subheader("Ground Truth (Stimulus)")
                if real_img is not None:
                    st.image(real_img, use_container_width=True)
                    st.caption(f"Path: `{row.get('folder', '')}/{row.get('filename', '')}`")
                else:
                    st.warning(f"Original image not found at:\n`{real_img_path}`")
                    
            with col_img2:
                st.subheader("AI Reconstruction")
                if recon_img is not None:
                    st.image(recon_img, use_container_width=True)
                    st.caption(f"Path: `.../{os.path.basename(os.path.dirname(os.path.dirname(recon_img_path)))}/{os.path.basename(os.path.dirname(recon_img_path))}/{os.path.basename(recon_img_path)}`")
                else:
                    st.warning(f"Reconstruction not found at:\n`{recon_img_path}`\n\nEnsure you have run the selected generation script for this split.")
                    
            # 5. Live Metrics Calculation
            if real_img is not None and recon_img is not None:
                st.divider()
                st.subheader("Step 07: Pixel Metrics")
                st.caption("Calculates MSE, SSIM, and PSNR on-the-fly, replicating `07_pixel_metrics.py`.")
                
                try:
                    # Calculate Metrics
                    mse = mean_squared_error(real_img.flatten(), recon_img.flatten())
                    ssim_val = ssim(real_img, recon_img, channel_axis=2, data_range=255)
                    
                    # Calculate PSNR safely
                    if mse == 0:
                        psnr_val = float('inf')
                    else:
                        psnr_val = 20 * np.log10(255.0 / np.sqrt(mse))
                    
                    # Display metrics as dashboard cards
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Mean Squared Error (MSE)", f"{mse:.2f}", help="Lower is better.")
                    m2.metric("Structural Similarity (SSIM)", f"{ssim_val:.4f}", help="Higher is better (Closer to 1.0).")
                    
                    psnr_str = "∞" if np.isinf(psnr_val) else f"{psnr_val:.2f} dB"
                    m3.metric("Peak Signal-to-Noise Ratio (PSNR)", psnr_str, help="Higher is better.")
                    
                except Exception as e:
                    st.error(f"Error calculating metrics: {e}")
                    
    else:
        st.error(f"❌ Split CSV not found: `{csv_path}`. Make sure you have run the Data Splitter script.")