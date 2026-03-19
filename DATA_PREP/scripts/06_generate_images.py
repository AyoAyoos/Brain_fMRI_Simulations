import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline

# ================= CONFIGURATION =================
# Path to the predictions
MODELS_DIR = r"D:/PBL_6/DATA_prep/output/models"
OUTPUT_DIR = r"D:/PBL_6/DATA_prep/output/reconstructions"

# Start with Objects
#INPUT_FILE = r"D:/PBL_6/DATA_prep/output/models/pred_objects.npy" 
INPUT_FILE = r"D:/PBL_6/DATA_prep/output/models/pred_scenes.npy"  # <-- Swap this later for the second run

# We'll use a standard SD 1.5 model and inject embeddings
MODEL_ID = "runwayml/stable-diffusion-v1-5"
# =================================================

def generate():
    # 1. Setup Output Folders
    file_label = INPUT_FILE.replace(".npy", "")
    save_folder = os.path.join(OUTPUT_DIR, file_label)
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 2. Load the Brain Predictions
    pred_file = os.path.join(MODELS_DIR, INPUT_FILE)
    print(f"⏳ Loading predictions from {pred_file}...")
    try:
        # Load (N, 1024) matrix
        embeddings = np.load(pred_file)
        print(f"   Loaded shape: {embeddings.shape}")
        
        # Convert to Tensor
        embeddings = torch.tensor(embeddings).float()
        
    except FileNotFoundError:
        print(f"❌ Error: Prediction file not found at {pred_file}")
        return

    # 3. Load Pipeline
    print(f"⏳ Loading Stable Diffusion ({MODEL_ID})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")
    
    # Determine dtype based on device
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe = pipe.to(device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nTrying alternative approach...")
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe = pipe.to(device)
        dtype = torch.float32
    
    # Get the text encoder's hidden size
    text_encoder_dim = pipe.text_encoder.config.hidden_size
    print(f"   Text encoder dimension: {text_encoder_dim}")
    print(f"   Your embedding dimension: {embeddings.shape[-1]}")

    # 4. Create a projection layer if dimensions don't match
    if embeddings.shape[-1] != text_encoder_dim:
        print(f"   Creating projection layer: {embeddings.shape[-1]} -> {text_encoder_dim}")
        projection = torch.nn.Linear(embeddings.shape[-1], text_encoder_dim)
        projection = projection.to(device).to(dtype)  # Match device AND dtype
        # Simple initialization
        torch.nn.init.xavier_uniform_(projection.weight)
    else:
        projection = None

    # 5. The Generation Loop
    num_to_generate = min(100, len(embeddings))
    print(f"\n🎨 Generating {num_to_generate} images...")
    print(f"   This will take approximately {num_to_generate * 0.5:.1f} minutes on GPU...")
    print(f"   Using dtype: {dtype}")
    
    for i in range(num_to_generate):
        if i % 10 == 0:
            print(f"   - Processing images {i}-{min(i+10, num_to_generate)} of {num_to_generate}...")
        
        # Get single brain signal: Shape (1024,)
        brain_signal = embeddings[i].unsqueeze(0).to(device).to(dtype)  # (1, 1024) - match dtype!
        
        # Project if needed
        if projection is not None:
            brain_signal = projection(brain_signal)  # (1, text_encoder_dim)
        
        # Reshape to match text encoder output: (batch, seq_len, hidden_dim)
        # Most SD models expect sequence length of 77
        brain_signal = brain_signal.unsqueeze(1)  # (1, 1, hidden_dim)
        brain_signal = brain_signal.repeat(1, 77, 1)  # (1, 77, hidden_dim)
        
        # Generate image
        try:
            with torch.no_grad():
                # Bypass text encoding and inject embeddings directly
                output = pipe(
                    prompt_embeds=brain_signal,
                    negative_prompt_embeds=None,
                    guidance_scale=7.5,
                    num_inference_steps=50,
                    height=512,
                    width=512
                )
                image = output.images[0]
            
            # Save
            output_path = os.path.join(save_folder, f"recon_{i:03d}.png")
            image.save(output_path)
            
        except Exception as e:
            print(f"\n   ⚠️ Error generating image {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n✅ Done! Check the folder: {save_folder}")
    print(f"   Generated {num_to_generate} reconstructions")

if __name__ == "__main__":
    generate()