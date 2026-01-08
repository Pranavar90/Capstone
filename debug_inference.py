
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import sys
import os
from scipy.signal import find_peaks
from sklearn.cluster import KMeans

# Add training dir to path
sys.path.append('training')
from model import DehazingModel

def estimate_haze_level_debug(image_pil):
    """Debug version of haze estimation."""
    img_np = np.array(image_pil)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # 1. Histogram
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist_smooth = np.convolve(hist, np.ones(5)/5, mode='same')
    peaks, _ = find_peaks(hist_smooth, distance=20, prominence=10)
    
    print(f"Debug: Found {len(peaks)} peaks in histogram.")
    
    n_clusters = len(peaks)
    if n_clusters < 1: n_clusters = 1
    
    # 2. K-means
    pixels = gray.reshape(-1, 1)
    if len(pixels) > 5000:
        indices = np.random.choice(len(pixels), 5000, replace=False)
        pixels_sample = pixels[indices]
    else:
        pixels_sample = pixels
        
    try:
        kmeans = KMeans(n_clusters=n_clusters, n_init=3, random_state=42)
        kmeans.fit(pixels_sample)
        centers = sorted(kmeans.cluster_centers_.flatten())
        print(f"Debug: Cluster centers: {centers}")
    except Exception as e:
        print(f"Debug: KMeans failed: {e}")
        centers = [np.mean(pixels)]
        
    # 3. Classify
    val = centers[-1] / 255.0
    print(f"Debug: Brightest center normalized: {val:.4f}")
    
    if val > 0.7:
        level = 2
    elif val > 0.4:
        level = 1
    else:
        level = 0
        
    print(f"Debug: Detected Haze Level: {level}")
    
    one_hot = torch.zeros(1, 3)
    one_hot[0, level] = 1.0
    return one_hot, level

def debug_inference(image_path, model_path):
    print(f"--- Debugging Image: {image_path} ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load Model
    model = DehazingModel()
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Load Image
    try:
        img_pil = Image.open(image_path).convert('RGB')
        print(f"Original Size: {img_pil.size}")
    except:
        print("Could not open image.")
        return

    # Estimate Haze
    condition, level = estimate_haze_level_debug(img_pil)
    condition = condition.to(device)

    # Transform
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(img_pil).unsqueeze(0).to(device)
    print(f"Input Tensor Mean: {input_tensor.mean():.4f}, Std: {input_tensor.std():.4f}")

    # Inference
    with torch.no_grad():
        output_tensor = model(input_tensor, condition)
    
    print(f"Output Tensor Mean: {output_tensor.mean():.4f}, Std: {output_tensor.std():.4f}")
    
    # Check L1 Distance (Input vs Output)
    l1_diff = torch.abs(input_tensor - output_tensor).mean().item()
    print(f"L1 Difference (Input vs Output): {l1_diff:.4f}")
    
    if l1_diff < 0.05:
        print("WARNING: Model output is very similar to input (Identity Mapping possibility).")
    else:
        print("Model is modifying the image significantly.")

    # Save output for inspection
    denorm = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    res = denorm(output_tensor.squeeze(0).cpu())
    res = torch.clamp(res, 0, 1)
    res_np = (res.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(res_np).save("debug_output.png")
    print("Saved debug_output.png")

if __name__ == "__main__":
    # Test on a generated hazy image
    test_img = "data/hazy/image_0000.jpg"
    if not os.path.exists(test_img):
        print(f"Test image {test_img} not found! Check data/hazy.")
    else:
        debug_inference(test_img, "models/dehaze_model_best.pth")
