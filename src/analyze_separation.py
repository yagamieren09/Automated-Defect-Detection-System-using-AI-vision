import os
import json
import torch
import sys
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Add patchcore to path
sys.path.append(os.path.abspath(os.path.join("patchcore-inspection", "src")))

from patchcore.patchcore import PatchCore
from patchcore.common import FaissNN

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "models/patchcore_tire"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VAL_GOOD_PATH = "dataset/mvtec_3d/tire/validation/good/rgb"
TEST_GOOD_PATH = "dataset/mvtec_3d/tire/test/good/rgb"
TEST_DEFECT_PATHS = {
    "cut": "dataset/mvtec_3d/tire/test/cut/rgb",
    "contamination": "dataset/mvtec_3d/tire/test/contamination/rgb",
    "hole": "dataset/mvtec_3d/tire/test/hole/rgb",
    "combined": "dataset/mvtec_3d/tire/test/combined/rgb"
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_scores(model, path):
    if not os.path.exists(path):
        return []
    files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg'))]
    scores = []
    for f in tqdm(files, desc=f"Scoring {os.path.basename(os.path.dirname(path))}", leave=False):
        img = Image.open(os.path.join(path, f)).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            score_list, _ = model.predict(img_t)
            score_val = score_list[0]
            if isinstance(score_val, torch.Tensor):
                raw_score = score_val.item()
            else:
                raw_score = float(score_val)
            scores.append(raw_score)
    return scores

def main():
    print("Loading PatchCore model...")
    model = PatchCore(device=DEVICE)
    model.load_from_path(load_path=MODEL_PATH, device=DEVICE, nn_method=FaissNN(False, 4))
    
    print("\nCollecting scores...")
    val_good = get_scores(model, VAL_GOOD_PATH)
    test_good = get_scores(model, TEST_GOOD_PATH)
    
    def_results = {}
    for name, path in TEST_DEFECT_PATHS.items():
        def_results[name] = get_scores(model, path)
    
    all_defects = [s for sublist in def_results.values() for s in sublist]

    print("\n" + "="*50)
    print("DISTRIBUTION ANALYSIS (RAW SCORES)")
    print("="*50)
    
    def print_stats(name, scores):
        if not scores: return
        scores = np.array(scores)
        print(f"{name:<15}: Count={len(scores)}, Min={scores.min():.4f}, Max={scores.max():.4f}, Mean={scores.mean():.4f}")
        print(f"{' '*16} p90={np.percentile(scores, 90):.4f}, p95={np.percentile(scores, 95):.4f}, p99={np.percentile(scores, 99):.4f}, p99.9={np.percentile(scores, 99.9):.4f}")

    print_stats("Good Val", val_good)
    print_stats("Good Test", test_good)
    print("-" * 50)
    for name, scores in def_results.items():
        print_stats(f"Defect: {name}", scores)
    print("-" * 50)
    print_stats("ALL DEFECTS", all_defects)

    # Calculate separation using Validation p99 as 1.0 (requested)
    if val_good:
        vp99 = np.percentile(val_good, 99)
        print(f"\nScaling with Validation p99 = {vp99:.4f} (mapping to 1.0)")
        
        def check_thresh(t):
            fp = sum(1 for s in test_good if s/vp99 >= t)
            tp = sum(1 for s in all_defects if s/vp99 >= t)
            print(f"Threshold {t:.2f}: FP={fp}/{len(test_good)} ({fp/len(test_good)*100:.1f}%), TP={tp}/{len(all_defects)} ({tp/len(all_defects)*100:.1f}%)")

        print("\nThreshold Sweep (Normalized by Validation p99):")
        for t in [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.20]:
            check_thresh(t)

if __name__ == "__main__":
    main()
