import sys
import os
import glob
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score

# --- CONFIGURATION ---
DATASET_ROOT = "dataset/mvtec_3d/tire"
MODEL_PATH = "models/patchcore_tire"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESOLUTION = 448

# --- SETUP PATHS ---
sys.path.append(os.path.abspath(os.path.join("patchcore-inspection", "src")))
from patchcore.patchcore import PatchCore
from patchcore.backbones import load as load_backbone
from patchcore.common import FaissNN

def load_model(save_path, device):
    """Load PatchCore model."""
    backbone = load_backbone("wideresnet50")
    backbone.name = "wideresnet50"
    model = PatchCore(device=device)
    model.load_from_path(save_path, device=device, nn_method=FaissNN(False, 4))
    return model

def get_image_transform(resolution):
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def main():
    print(f"Loading PatchCore model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH, DEVICE)
    transform = get_image_transform(RESOLUTION)

    # --- STEP 1: Calibrating on GOOD Validation ---
    val_good_files = glob.glob(os.path.join(DATASET_ROOT, "validation/good/rgb/*.png"))
    val_scores = []
    with torch.no_grad():
        for f in val_good_files:
            img = Image.open(f).convert("RGB")
            img_t = transform(img).unsqueeze(0).to(DEVICE)
            score, _ = model.predict(img_t)
            s = score[0].item() if hasattr(score[0], 'item') else score[0]
            val_scores.append(s)
            
    val_min, val_max = np.min(val_scores), np.max(val_scores)
    # Norm such that Max GOOD is 0.58
    norm_denom = (val_max - val_min) / 0.58 if (val_max - val_min) > 0 else 1.0

    # --- STEP 2: Collecting Inference Scores ---
    test_root = os.path.join(DATASET_ROOT, "test")
    categories = ["good", "cut", "contamination", "hole", "combined"]
    
    all_scores = []
    all_labels = [] # 0 for good, 1 for defect
    all_results = []

    for cat in categories:
        cat_path = os.path.join(test_root, cat, "rgb")
        if not os.path.exists(cat_path): cat_path = os.path.join(test_root, cat)
        files = glob.glob(os.path.join(cat_path, "*.png"))
        if not files: continue
        
        for f in sorted(files):
            with torch.no_grad():
                img = Image.open(f).convert("RGB")
                img_t = transform(img).unsqueeze(0).to(DEVICE)
                score, _ = model.predict(img_t)
                raw = score[0].item() if hasattr(score[0], 'item') else score[0]
                norm = (raw - val_min) / norm_denom
                
                label = 0 if cat == 'good' else 1
                all_scores.append(norm)
                all_labels.append(label)
                all_results.append({'file': os.path.basename(f), 'cat': cat, 'norm': norm, 'label': label})

    # --- STEP 3: Metrics Calculation ---
    GLOBAL_THRESHOLD = 0.60
    
    # Decisions based on GLOBAL THRESHOLD ONLY
    preds = [1 if s > GLOBAL_THRESHOLD else 0 for s in all_scores]
    accuracy = sum(1 for p, l in zip(preds, all_labels) if p == l) / len(all_labels)
    
    # Defect detection rate (Recall of all defects)
    defect_indices = [i for i, l in enumerate(all_labels) if l == 1]
    defect_preds = [preds[i] for i in defect_indices]
    detection_rate = sum(defect_preds) / len(defect_preds) if defect_preds else 0
    
    # AUROC
    auroc = roc_auc_score(all_labels, all_scores)

    # --- STEP 4: Reporting ---
    print("\n" + "="*60)
    print("UNIFIED INFERENCE BASELINE (No Overrides)")
    print("="*60)
    print(f"Global Threshold:     {GLOBAL_THRESHOLD:.2f}")
    print(f"Total Test Images:    {len(all_labels)}")
    print(f"Overall Accuracy:     {accuracy*100:.2f}%")
    print(f"Defect Detection:     {detection_rate*100:.2f}%")
    print(f"Overall AUROC:        {auroc:.4f}")
    print("="*60)

    # Per-category summary
    summary = {c: {'total': 0, 'correct': 0, 'reject': 0} for c in categories}
    for r in all_results:
        decision = "REJECT" if r['norm'] > GLOBAL_THRESHOLD else "ACCEPT"
        summary[r['cat']]['total'] += 1
        is_correct = (decision == "REJECT" and r['label'] == 1) or (decision == "ACCEPT" and r['label'] == 0)
        if is_correct: summary[r['cat']]['correct'] += 1
        if decision == "REJECT": summary[r['cat']]['reject'] += 1

    print(f"\n{'Category':<15} | {'Total':<6} | {'Correct':<8} | {'Recall/Acc':<10}")
    print("-" * 50)
    for cat in categories:
        s = summary[cat]
        if s['total'] == 0: continue
        rate = s['correct'] / s['total']
        print(f"{cat:<15} | {s['total']:<6} | {s['correct']:<8} | {rate*100:.2f}%")

if __name__ == "__main__":
    main()
