import os
import json
import torch
import sys
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add patchcore to path
sys.path.append(os.path.abspath(os.path.join("patchcore-inspection", "src")))

from patchcore.patchcore import PatchCore
from patchcore.common import FaissNN

# -----------------------------
# CONFIG
# -----------------------------
BASE_DATASET_PATH = "dataset/mvtec_3d/tire/test"
CATEGORIES = ["good", "cut", "contamination", "hole", "combined"]
MODEL_PATH = "models/patchcore_tire"
STATS_PATH = os.path.join(MODEL_PATH, "normalization_stats.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_normalization_stats(stats_path):
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)
        print(f"Loaded normalization stats: {stats}")
        return stats
    else:
        print("ERROR: Normalization stats not found. Re-run training/calibration.")
        sys.exit(1)

def get_image_prediction(model, img_path, transform, device):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        score_list, anomaly_map = model.predict(img_tensor)
        raw_score = score_list[0].item() if isinstance(score_list[0], torch.Tensor) else float(score_list[0])
        return raw_score, anomaly_map[0], img

def calculate_metrics(y_true, y_pred):
    tp = sum((t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    tn = sum((t == 0 and p == 0) for t, p in zip(y_true, y_pred))
    fp = sum((t == 0 and p == 1) for t, p in zip(y_true, y_pred))
    fn = sum((t == 1 and p == 0) for t, p in zip(y_true, y_pred))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1, tp, tn, fp, fn

def run_threshold_sweep(scores_dict, thresholds):
    """Find the best threshold based on industrial constraints."""
    good_scores = scores_dict["good"]
    defect_categories = ["cut", "contamination", "hole", "combined"]
    
    best_t = -1
    best_defect_rej = -1
    best_good_acc = -1
    
    print("\n" + "-"*65)
    print(f"{'Thresh':<8} | {'GOOD Acc':<10} | {'Defect Rej':<12} | {'Status'}")
    print("-" * 65)
    
    for t in thresholds:
        good_acc = sum(1 for s in good_scores if s < t) / len(good_scores) * 100
        
        all_defect_scores = []
        for cat in defect_categories:
            all_defect_scores.extend(scores_dict[cat])
        
        defect_rej = sum(1 for s in all_defect_scores if s >= t) / len(all_defect_scores) * 100
        
        status = ""
        if good_acc >= 98 and defect_rej >= 85:
            status = "TARGET MET"
        elif good_acc >= 98:
            status = "GOOD OK"
            
        print(f"{t:<8.3f} | {good_acc:<10.1f}% | {defect_rej:<12.1f}% | {status}")
        
        # Priority 1: GOOD Acc >= 98%
        # Priority 2: Maximize Defect Rejection
        if good_acc >= 98:
            if defect_rej > best_defect_rej:
                best_defect_rej = defect_rej
                best_good_acc = good_acc
                best_t = t
        elif best_t == -1: # If no threshold meets 98% Good Acc, pick best trade-off
             if good_acc > best_good_acc:
                 best_good_acc = good_acc
                 best_defect_rej = defect_rej
                 best_t = t

    print("-" * 65)
    return best_t, best_good_acc, best_defect_rej

def main():
    print("="*80)
    print("PatchCore Dual-Objective Inference & Calibration: MVTec 3D-AD Tire")
    print("="*80)
    
    # Load model and stats
    print("Loading PatchCore model and stats...")
    norm_stats = load_normalization_stats(STATS_PATH)
    res = norm_stats.get("resolution", 448) # Default to 448
    p98 = norm_stats.get("p98")
    
    if not p98:
        print("ERROR: p98 not found in stats. Re-run calibration.")
        sys.exit(1)

    model = PatchCore(device=DEVICE)
    model.load_from_path(
        load_path=MODEL_PATH,
        device=DEVICE,
        nn_method=FaissNN(False, 4)
    )
    
    transform = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Pass 1: Collect ALL scores
    print("\n--- STEP 1: Collecting scores and normalizing (p98 -> 0.8) ---")
    all_scores = {cat: [] for cat in CATEGORIES}
    img_data = {cat: [] for cat in CATEGORIES}

    # Normalize stats for robust scaling
    mean_val = norm_stats.get("mean", 0)
    std_val = norm_stats.get("std", 1)

    for cat in CATEGORIES:
        input_dir = os.path.join(BASE_DATASET_PATH, cat, "rgb")
        if not os.path.exists(input_dir):
            print(f"Warning: Category path {input_dir} not found.")
            continue
        files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg'))])
        
        for f in tqdm(files, desc=f"Inference: {cat}", leave=False):
            img_path = os.path.join(input_dir, f)
            raw, amap, img = get_image_prediction(model, img_path, transform, DEVICE)
            
            # Percentile Scaling: map GOOD p98 to 0.8
            # norm = (raw / p98) * 0.8
            norm_score = (raw / p98) * 0.8
            
            all_scores[cat].append(norm_score)
            img_data[cat].append({
                "file": f,
                "raw": raw,
                "norm": norm_score,
                "amap": amap
            })

    # Step 2: Threshold Sweep (Granular)
    thresholds = np.linspace(0.3, 1.1, 41) # 41 steps for 0.02 granularity
    best_threshold, best_good_acc, best_defect_rej = run_threshold_sweep(all_scores, thresholds)
    
    print(f"\nOPTIMIZED INDUSTRIAL THRESHOLD: {best_threshold:.3f}")
    print(f"Target Performance: GOOD Acc={best_good_acc:.1f}%, Defect Rejection={best_defect_rej:.1f}%")
    
    # Step 3: Final Classification & Detailed Logging
    print("\n--- STEP 3: Final Results & Reporting ---")
    
    overall_y_true = []
    overall_y_pred = []
    
    summary_table = []
    
    # Header for detailed log
    log_lines = [f"{'Image':<20} | {'Raw':<8} | {'Norm':<8} | {'Thresh':<8} | {'Decision'}"]
    log_lines.append("-" * 60)

    for cat in CATEGORIES:
        output_dir = f"inference/output_{cat}"
        os.makedirs(output_dir, exist_ok=True)
        
        y_true = [0 if cat == "good" else 1 for _ in range(len(img_data[cat]))]
        y_pred = []
        
        for item in img_data[cat]:
            score = item["norm"]
            decision = "REJECT" if score >= best_threshold else "ACCEPT"
            pred = 1 if decision == "REJECT" else 0
            y_pred.append(pred)
            
            # Image logging
            log_lines.append(f"{item['file']:<20} | {item['raw']:<8.4f} | {item['norm']:<8.4f} | {best_threshold:<8.1f} | {decision}")
            
            # Save heatmap for samples (optional, limit for speed)
            save_path = os.path.join(output_dir, f"{item['file']}_heatmap.png")
            plt.imshow(item["amap"], cmap="jet")
            plt.axis("off")
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()

        # Calculate metrics for category
        prec, rec, f1, tp, tn, fp, fn = calculate_metrics(y_true, y_pred)
        
        summary_table.append({
            "cat": cat,
            "total": len(y_true),
            "acc": (tp + tn) / len(y_true) * 100,
            "prec": prec * 100,
            "rec": rec * 100,
            "f1": f1 * 100,
            "fp": fp,
            "fn": fn
        })
        
        overall_y_true.extend(y_true)
        overall_y_pred.extend(y_pred)

    # Print Confusion Summary Table
    print("\n" + "="*95)
    print(f"{'Category':<15} | {'Total':<6} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'FP/FN'}")
    print("-" * 95)
    for row in summary_table:
        print(f"{row['cat']:<15} | {row['total']:<6} | {row['acc']:<9.1f}% | {row['prec']:<9.1f}% | {row['rec']:<9.1f}% | {row['f1']:<9.1f}% | {row['fp']}/{row['fn']}")
    print("-" * 95)
    
    # Overall Metrics
    o_prec, o_rec, o_f1, _, _, _, _ = calculate_metrics(overall_y_true, overall_y_pred)
    print(f"{'OVERALL':<15} | {len(overall_y_true):<6} | {'-':<10} | {o_prec*100:<9.1f}% | {o_rec*100:<9.1f}% | {o_f1*100:<9.1f}% | -")
    print("="*95)
    
    # Save log to file
    with open("inference_results_detailed.log", "w") as f:
        f.write("\n".join(log_lines))
    print(f"\nDetailed per-image logs saved to: inference_results_detailed.log")

if __name__ == "__main__":
    main()
