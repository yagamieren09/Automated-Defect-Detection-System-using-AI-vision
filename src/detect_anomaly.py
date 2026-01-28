import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Add patchcore to path
import sys
sys.path.append("patchcore-inspection/src")

from patchcore.patchcore import PatchCore
from patchcore.common import FaissNN
from patchcore.backbones import load as load_backbone


print("Loading PatchCore model...")

# -----------------------------
# CONFIG
# -----------------------------
DATASET_PATH = "dataset/mvtec_3d/tire/test"
OUTPUT_PATH = "inference/output_images"
MODEL_PATH = "models/patchcore_tire"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
model = PatchCore(device=DEVICE)
model.load_from_path(
    load_path=MODEL_PATH,
    device=DEVICE,
    nn_method=FaissNN(False, 4)
)

print("Model loaded successfully.")

# -----------------------------
# RUN INFERENCE
# -----------------------------
print("Running inference...")

for defect_type in os.listdir(DATASET_PATH):
    defect_folder_root = os.path.join(DATASET_PATH, defect_type)
    defect_folder = os.path.join(defect_folder_root, "rgb")

    if not os.path.isdir(defect_folder):
        continue

    print(f"Processing {defect_type}...")
    for img_name in os.listdir(defect_folder):
        img_path = os.path.join(defect_folder, img_name)

        if not os.path.isfile(img_path) or not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            score, anomaly_map = model.predict(img_tensor)
            score = score[0]
            anomaly_map = anomaly_map[0]

        THRESHOLD = 0.30   # tuned for MVTec 3D tire
        decision = "REJECT" if score > THRESHOLD else "ACCEPT"


        print(f"{img_name} | Defect: {defect_type} | Score: {score:.3f} | {decision}")

        # Save heatmap
        plt.imshow(anomaly_map, cmap="jet")
        plt.axis("off")
        save_path = os.path.join(
            OUTPUT_PATH, f"{img_name}_heatmap.png"
        )
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

print(f"Using anomaly threshold = {THRESHOLD}")

print("Inference completed.")
