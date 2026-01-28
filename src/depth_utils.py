import numpy as np
import cv2
import torch
from PIL import Image
import tifffile as tiff

def load_depth_image(path):
    """
    Loads a depth image from a path. 
    Supports .tiff for XYZ (MVTec 3D-AD format).
    Returns the Z channel as a numpy array.
    """
    if path.lower().endswith(('.tiff', '.tif')):
        # MVTec 3D-AD XYZ images are 3-channel TIFFs (X, Y, Z)
        # We only need the Z channel (channel 2)
        try:
            image = tiff.imread(path)
            # XYZ is typically (H, W, 3)
            if len(image.shape) == 3 and image.shape[2] == 3:
                z_channel = image[:, :, 2]
                return z_channel
        except Exception as e:
            print(f"Error loading TIFF {path}: {e}")
            return None
    
    # Fallback for png/jpg depth maps if they exist
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

def preprocess_depth_map(depth_map, target_resolution=224):
    """
     Cleans, normalizes, and resizes a depth map.
     1. Remove NaNs/Infs (replace with min valid value)
     2. Clip outliers (p1 - p99)
     3. Normalize to [0, 1]
     4. Convert to 3-channel (R=G=B)
     5. Resize to target_resolution
    """
    if depth_map is None:
        return None
        
    # Handle NaNs and Infs
    # Replace NaNs with the minimum valid finite value to avoid artifacts
    mask = np.isfinite(depth_map)
    if not np.any(mask):
        # Image is all NaN/Inf - return zero image
        depth_map = np.zeros_like(depth_map)
    else:
        min_valid = np.min(depth_map[mask])
        depth_map = np.nan_to_num(depth_map, nan=min_valid, posinf=min_valid, neginf=min_valid)

    # Clip outliers (p1 - p99) to remove noise spikes
    p1 = np.percentile(depth_map, 1)
    p99 = np.percentile(depth_map, 99)
    depth_map = np.clip(depth_map, p1, p99)

    # Normalize to [0, 1]
    d_min = depth_map.min()
    d_max = depth_map.max()
    if d_max - d_min > 1e-8:
        depth_map = (depth_map - d_min) / (d_max - d_min)
    else:
        depth_map = np.zeros_like(depth_map) # uniform depth

    # Convert to 8-bit for resizing/tensor conversion
    depth_map_uint8 = (depth_map * 255).astype(np.uint8)
    
    # Resize
    depth_map_resized = cv2.resize(depth_map_uint8, (target_resolution, target_resolution), interpolation=cv2.INTER_AREA)

    # Convert to 3-channel (duplicate channels) since backbone expects RGB
    depth_3ch = np.stack([depth_map_resized] * 3, axis=-1)
    
    return depth_3ch

def get_depth_transform(resolution):
    """
    Returns a transform-like callable that processes a raw depth path into a tensor.
    """
    from torchvision import transforms
    
    # Standard ImageNet normalization for WideResNet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def transform_fn(img_path):
        # 1. Load raw Z-channel
        raw_depth = load_depth_image(img_path)
        
        # 2. Preprocess (clean, norm, resize, 3-channel)
        processed_depth = preprocess_depth_map(raw_depth, target_resolution=resolution)
        
        # 3. To Tensor
        # PIL Image -> ToTensor -> Normalize
        pil_img = Image.fromarray(processed_depth)
        tensor_img = transforms.ToTensor()(pil_img)
        norm_img = normalize(tensor_img)
        
        return norm_img

    return transform_fn
