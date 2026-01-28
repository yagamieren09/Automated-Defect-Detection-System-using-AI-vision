import os
import json
import torch
import sys
from PIL import Image
from torchvision import transforms

# Add patchcore to path
sys.path.append(os.path.abspath(os.path.join("patchcore-inspection", "src")))

from patchcore.patchcore import PatchCore
from patchcore.common import FaissNN

# -----------------------------
# CONFIG
# -----------------------------
VALIDATION_PATH = "dataset/mvtec_3d/tire/validation/good/rgb"
MODEL_PATH = "models/patchcore_tire"
STATS_PATH = os.path.join(MODEL_PATH, "normalization_stats.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# IMAGE TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def main():
    print("Loading PatchCore model...")
    model = PatchCore(device=DEVICE)
    model.load_from_path(
        load_path=MODEL_PATH,
        device=DEVICE,
        nn_method=FaissNN(False, 4)
    )
    print("Model loaded successfully.")

    if not os.path.exists(VALIDATION_PATH):
        print(f"Error: Validation path not found: {VALIDATION_PATH}")
        return

    print(f"Running calibration on: {VALIDATION_PATH}")
    scores = []
    
    image_files = sorted([f for f in os.listdir(VALIDATION_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not image_files:
        print("No validation images found!")
        return

    for img_name in image_files:
        img_path = os.path.join(VALIDATION_PATH, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                score_list, _ = model.predict(img_tensor)
                # model.predict returns a list/tensor of scores. Get the first one.
                val = score_list[0]
                # If it's a tensor, get the float value
                if isinstance(val, torch.Tensor):
                    val = val.item()
                scores.append(float(val))
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    if scores:
        scores = np.array(scores)
        min_score = float(scores.min())
        max_score = float(scores.max())
        mean_score = float(scores.mean())
        std_score = float(scores.std())
        
        # Calculate robust percentiles
        p95 = float(np.percentile(scores, 95))
        p96 = float(np.percentile(scores, 96))
        p97 = float(np.percentile(scores, 97))
        p98 = float(np.percentile(scores, 98))
        p99 = float(np.percentile(scores, 99))
        
        # Get resolution from training config if possible
        # Default to 448 as per train_patchcore.py
        resolution = 448 
        
        stats = {
            "min": min_score,
            "max": max_score,
            "mean": mean_score,
            "std": std_score,
            "p95": p95,
            "p96": p96,
            "p97": p97,
            "p98": p98,
            "p99": p99,
            "resolution": resolution
        }
        
        with open(STATS_PATH, "w") as f:
            json.dump(stats, f, indent=4)
            
        print("-" * 40)
        print(f"Calibration Complete.")
        print(f"Processed {len(scores)} images.")
        print(f"Mean Score: {mean_score:.4f} (std: {std_score:.4f})")
        print(f"p96: {p96:.4f}, p98: {p98:.4f}, p99: {p99:.4f}")
        print(f"Stats saved to: {STATS_PATH}")
        print("-" * 40)
    else:
        print("No scores computed.")

if __name__ == "__main__":
    import numpy as np
    main()
