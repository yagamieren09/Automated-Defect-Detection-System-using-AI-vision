import sys
import os
import json
import torch
import numpy as np
from tqdm import tqdm

# Add patchcore-inspection to path
sys.path.append(os.path.abspath(os.path.join("patchcore-inspection", "src")))

from patchcore.patchcore import PatchCore
from patchcore.backbones import load as load_backbone
from patchcore.sampler import ApproximateGreedyCoresetSampler
from patchcore.common import FaissNN

# Import our new depth utils
import depth_utils

# ----------------------------
# CONFIG
# ----------------------------
# Note: Depth paths often mirror RGB paths but in 'xyz' folder
RGB_TRAIN_PATH = "dataset/mvtec_3d/tire/train/good/rgb" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESOLUTION = 448
SAVE_PATH = "models/patchcore_tire_depth"

def main():
    print("="*80)
    print("PatchCore Training: DEPTH Stream")
    print("="*80)

    # ----------------------------
    # LOAD MODEL
    # ----------------------------
    print(f"Loading backbone for {RESOLUTION}x{RESOLUTION} resolution...")
    backbone = load_backbone("wideresnet50")
    backbone.name = "wideresnet50"

    patchcore = PatchCore(device=DEVICE)
    patchcore.load(
        backbone=backbone,
        layers_to_extract_from=["layer2", "layer3"],
        device=DEVICE,
        input_shape=(3, RESOLUTION, RESOLUTION),
        pretrain_embed_dimension=1024,
        target_embed_dimension=384,
        patchsize=3,
        patchstride=1,
        anomaly_scorer_num_nn=1,
        featuresampler=ApproximateGreedyCoresetSampler(percentage=0.01, device=DEVICE),
        nn_method=FaissNN(False, 4),
    )
    print("PatchCore model loaded.")

    # ----------------------------
    # LOAD DEPTH IMAGES
    # ----------------------------
    print("Loading GOOD DEPTH images...")
    
    # Helper to convert RGB path to XYZ path
    # RGB: .../tire/train/good/rgb/000.png
    # XYZ: .../tire/train/good/xyz/000.tiff
    def get_xyz_path(rgb_path):
        xyz_path = rgb_path.replace("rgb", "xyz").replace(".png", ".tiff")
        return xyz_path

    transform_fn = depth_utils.get_depth_transform(RESOLUTION)
    images = []

    if not os.path.exists(RGB_TRAIN_PATH):
        print(f"Error: Training path {RGB_TRAIN_PATH} not found.")
        sys.exit(1)

    file_list = sorted([f for f in os.listdir(RGB_TRAIN_PATH) if f.endswith(".png")])
    
    for filename in tqdm(file_list, desc="Loading Depth Data"):
        rgb_full_path = os.path.join(RGB_TRAIN_PATH, filename)
        xyz_full_path = get_xyz_path(rgb_full_path)
        
        if os.path.exists(xyz_full_path):
            try:
                img_tensor = transform_fn(xyz_full_path)
                images.append(img_tensor)
            except Exception as e:
                print(f"Failed to process {xyz_full_path}: {e}")
        else:
            print(f"Warning: Missing depth file {xyz_full_path}")

    if not images:
        print("No valid depth images found. Aborting.")
        sys.exit(1)

    images = torch.stack(images).to(DEVICE)
    print(f"Loaded {len(images)} good depth images.")

    # ----------------------------
    # TRAINING (FEATURE MEMORY)
    # ----------------------------
    print("Starting full PatchCore training (Depth)...")
    dataloader = torch.utils.data.DataLoader(images, batch_size=4)
    patchcore.fit(dataloader)

    # ----------------------------
    # SAVE MODEL
    # ----------------------------
    os.makedirs(SAVE_PATH, exist_ok=True)
    patchcore.save_to_path(SAVE_PATH)
    print(f"Depth model saved to {SAVE_PATH}")

    # ----------------------------
    # CALIBRATION (VALIDATION)
    # ----------------------------
    # We do calibration on the FUSED score later in inference, 
    # but we can optionally save some depth-only stats here if needed.
    # For now, we will skip depth-only calibration saving as we rely on the unified pipeline.
    
    print("Training complete.")

if __name__ == "__main__":
    main()
