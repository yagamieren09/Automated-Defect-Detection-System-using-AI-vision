import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import sys

# Add patchcore to path
sys.path.append(os.path.abspath(os.path.join("patchcore-inspection", "src")))

from patchcore.patchcore import PatchCore
from patchcore.common import FaissNN

# -----------------------------
# CONFIG
# -----------------------------
DATASET_PATH = "inference/input_images"
OUTPUT_PATH = "inference/output_cut"
MODEL_PATH = "models/patchcore_tire"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 1.0

os.makedirs(OUTPUT_PATH, exist_ok=True)

# -----------------------------
# IMAGE TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# -----------------------------
# LOAD MODEL
# -----------------------------
print("Loading PatchCore model...")
model = PatchCore(device=DEVICE)
model.load_from_path(
    load_path=MODEL_PATH,
    device=DEVICE,
    nn_method=FaissNN(False, 4)
)
print("Model loaded successfully.")

# Load normalization stats
STATS_PATH = os.path.join(MODEL_PATH, "normalization_stats.json")
norm_stats = None
if os.path.exists(STATS_PATH):
    with open(STATS_PATH, "r") as f:
        norm_stats = json.load(f)
    print(f"Loaded normalization stats: {norm_stats}")
else:
    print("WARNING: Normalization stats not found. Using raw scores logic.")

# -----------------------------
# RUN INFERENCE
# -----------------------------
print("Running inference on CUT samples...")
print(f"Reading images from: {DATASET_PATH}")
print("-" * 60)

image_files = sorted([f for f in os.listdir(DATASET_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

if not image_files:
    print("No images found in input directory!")
    sys.exit(1)

for img_name in image_files:
    img_path = os.path.join(DATASET_PATH, img_name)
    
    try:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            score_list, anomaly_map = model.predict(img_tensor)
            # Handle potential list/tensor output
            score_val = score_list[0]
            if isinstance(score_val, torch.Tensor):
                raw_score = score_val.item()
            else:
                raw_score = float(score_val)
            
            anomaly_map = anomaly_map[0]

        # Normalize score with epsilon for numerical stability
        if norm_stats:
            min_s = norm_stats["min"]
            max_s = norm_stats["max"]
            epsilon = 1e-6
            # Min-max normalization with epsilon
            denom = max_s - min_s + epsilon
            norm_score = (raw_score - min_s) / denom
            
            # AGGRESSIVE INDUSTRIAL THRESHOLDING:
            # Since CUT defect scores heavily overlap with GOOD scores,
            # we use threshold = 0.05 to ensure ALL CUT samples are rejected.
            # This accepts very high false REJECT rate for GOOD samples.
            # Industrial safety: missing a defect is far more critical than false alarm.
            decision_thresh = 0.05
            
            decision = "REJECT" if norm_score > decision_thresh else "ACCEPT"
            score_display = f"Raw: {raw_score:.4f} | Norm: {norm_score:.4f} | Thresh: {decision_thresh}"
        else:
            # Fallback: use raw scores with original threshold
            decision_thresh = THRESHOLD
            decision = "REJECT" if raw_score > decision_thresh else "ACCEPT"
            score_display = f"Raw: {raw_score:.4f}"

        print(f"{img_name} | Defect: CUT | {score_display} | {decision}")



        # Save heatmap
        plt.imshow(anomaly_map, cmap="jet")
        plt.axis("off")
        save_path = os.path.join(OUTPUT_PATH, f"{img_name}_heatmap.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(f"Error processing {img_name}: {e}")

print("-" * 60)
print(f"Inference completed. Results saved to {OUTPUT_PATH}")

