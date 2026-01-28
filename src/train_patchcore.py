import sys
import os
import json

# Add patchcore-inspection to path
sys.path.append(os.path.abspath(os.path.join("patchcore-inspection", "src")))

import torch
from torchvision import transforms
from PIL import Image
from patchcore.patchcore import PatchCore
from patchcore.backbones import load as load_backbone
from patchcore.sampler import ApproximateGreedyCoresetSampler
from patchcore.common import FaissNN

# ----------------------------
# CONFIG
# ----------------------------
DATASET_PATH = "dataset/mvtec_3d/tire/train/good/rgb"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESOLUTION = 448

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
    # Trying small coreset ratio for better robustness/separation as per some PatchCore papers
    featuresampler=ApproximateGreedyCoresetSampler(percentage=0.01, device=DEVICE),
    nn_method=FaissNN(False, 4),
)

print("PatchCore model loaded.")

# ----------------------------
# IMAGE TRANSFORM
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((RESOLUTION, RESOLUTION)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ----------------------------
# LOAD GOOD IMAGES
# ----------------------------
print("Loading GOOD images...")
images = []

for file in os.listdir(DATASET_PATH):
    if file.endswith(".png") or file.endswith(".jpg"):
        img = Image.open(os.path.join(DATASET_PATH, file)).convert("RGB")
        img = transform(img)
        images.append(img)

images = torch.stack(images).to(DEVICE)
print(f"Loaded {len(images)} good images.")

# ----------------------------
# TRAINING (FEATURE MEMORY)
# ----------------------------
print("Starting full PatchCore training (448x448)...")
dataloader = torch.utils.data.DataLoader(images, batch_size=4) # Smaller batch for 448
patchcore.fit(dataloader)

# ----------------------------
# SAVE MODEL
# ----------------------------
SAVE_PATH = "models/patchcore_tire"
os.makedirs(SAVE_PATH, exist_ok=True)
patchcore.save_to_path(SAVE_PATH)

# ----------------------------
# CALIBRATION (VALIDATION)
# ----------------------------
import numpy as np
print("Running calibration on VALIDATION set...")
VALIDATION_PATH = "dataset/mvtec_3d/tire/validation/good/rgb"
validation_scores = []

if os.path.exists(VALIDATION_PATH):
    val_files = [f for f in os.listdir(VALIDATION_PATH) if f.endswith(('.png', '.jpg'))]
    
    for val_file in val_files:
        img_path = os.path.join(VALIDATION_PATH, val_file)
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            score_list, _ = patchcore.predict(img_tensor)
            val = score_list[0].item() if isinstance(score_list[0], torch.Tensor) else float(score_list[0])
            validation_scores.append(val)

    if validation_scores:
        validation_scores = np.array(validation_scores)
        min_score = float(validation_scores.min())
        max_score = float(validation_scores.max())
        p99 = float(np.percentile(validation_scores, 99))
        p99_9 = float(np.percentile(validation_scores, 99.9))
        
        stats = {
            "min": min_score,
            "max": max_score,
            "p99": p99,
            "p99.9": p99_9,
            "resolution": RESOLUTION
        }
        
        stats_path = os.path.join(SAVE_PATH, "normalization_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)
            
        print(f"Calibration complete. Stats saved to {stats_path}")
        print(f"Validation Min: {min_score:.4f}, Max: {max_score:.4f}, p99: {p99:.4f}")
    else:
        print("No validation images found. Skipping calibration.")
else:
    print(f"Validation path not found: {VALIDATION_PATH}")
    
print("Training complete. Model saved successfully.")
