import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from efficient_ad.model import get_pdn_small, get_autoencoder
from efficient_ad.dataset import TireDataset, get_transforms

def analyze(dataset_path, model_path, device='cpu'):
    # Load Model
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    ckpt = torch.load(model_path, map_location=device)
    teacher = get_pdn_small().to(device)
    student = get_pdn_small().to(device)
    autoencoder = get_autoencoder().to(device)
    
    teacher.load_state_dict(ckpt['teacher'])
    student.load_state_dict(ckpt['student'])
    autoencoder.load_state_dict(ckpt['autoencoder'])
    
    mu, sigma = ckpt['mu'].to(device), ckpt['sigma'].to(device)
    
    teacher.eval()
    student.eval()
    autoencoder.eval()
    
    # Setup Test Data
    image_size = ckpt.get('image_size', 128)
    test_transform = get_transforms(image_size=image_size, is_train=False)
    test_dataset = TireDataset(dataset_path, transform=test_transform, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    category_scores = {}
    
    print(f"Running analysis on {len(test_dataset)} images...")
    for images, categories, basenames in tqdm(test_loader):
        images = images.to(device)
        cat = categories[0]
        
        with torch.no_grad():
            t_out = (teacher(images) - mu) / (sigma + 1e-6)
            s_out = student(images)
            ae_out = autoencoder(images)
            
            map_st = torch.mean((t_out - s_out)**2, dim=1, keepdim=True)
            map_ae = torch.mean((t_out - ae_out)**2, dim=1, keepdim=True)
            
            score_st = torch.max(map_st).item()
            score_ae = torch.max(map_ae).item()
            score_comb = 0.5 * score_st + 0.5 * score_ae
            
        if cat not in category_scores:
            category_scores[cat] = []
        category_scores[cat].append({
            'st': score_st,
            'ae': score_ae,
            'comb': score_comb
        })
        
    # Print statistics
    print("\nScore Statistics by Category:")
    for cat, scores in category_scores.items():
        st_scores = [s['st'] for s in scores]
        ae_scores = [s['ae'] for s in scores]
        comb_scores = [s['comb'] for s in scores]
        
        print(f"{cat:15s} | Count: {len(scores):3d} | ST: {np.mean(st_scores):.4f} | AE: {np.mean(ae_scores):.4f} | Combined: {np.mean(comb_scores):.4f}")

    # Plot distribution
    plt.figure(figsize=(12, 6))
    for cat, scores in category_scores.items():
        comb_scores = [s['comb'] for s in scores]
        plt.hist(comb_scores, alpha=0.5, label=cat, bins=20)
    
    plt.title("Score Distribution by Category")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("anomaly_distribution.png")
    print("\nDistribution plot saved to anomaly_distribution.png")

if __name__ == "__main__":
    analyze("dataset/mvtec_3d/tire", "models/efficient_ad_tire.pth")
