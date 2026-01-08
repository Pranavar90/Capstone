"""
Inference module for image dehazing.
Handles model loading, preprocessing, and postprocessing.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import numpy as np
import cv2
from scipy.signal import find_peaks
from sklearn.cluster import KMeans

# Import model architecture
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training'))
from model import DehazingModel


class DehazingInference:
    """
    Inference wrapper for the dehazing model.
    Keeps model on GPU for fast inference.
    """
    
    def __init__(self, model_path, device='cuda'):
        """
        Initialize inference engine.
        """
        # Enforce CUDA requirement
        assert torch.cuda.is_available(), \
            "CUDA is required for inference! Please use a CUDA-capable GPU."
        
        self.device = torch.device(device)
        print(f"Initializing inference on {self.device}")
        
        # Load model
        self.model = DehazingModel()
        
        # Verify model exists
        if not os.path.exists(model_path):
             # Fallback paths
             if os.path.exists('../models/dehaze_model_best.pth'):
                 model_path = '../models/dehaze_model_best.pth'
             elif os.path.exists('models/dehaze_model_best.pth'):
                 model_path = 'models/dehaze_model_best.pth'
        
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing transform (Normalization only)
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Denormalization for output
        self.denormalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
    
    def estimate_haze_level(self, image_pil):
        """
        Estimate haze severity using Histogram Peaks + K-Means.
        Matches training logic.
        """
        # Resize to 512 for estimation consistency
        img_small = image_pil.resize((512, 512))
        img_np = np.array(img_small)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # 1. Histogram & Peaks
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist_smooth = np.convolve(hist, np.ones(5)/5, mode='same')
        peaks, _ = find_peaks(hist_smooth, distance=20, prominence=10)
        
        n_clusters = len(peaks)
        if n_clusters < 1: n_clusters = 1
        
        # 2. K-Means
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
        except:
            centers = [np.mean(pixels)]
            
        # 3. Classify
        val = centers[-1] / 255.0
        
        if val > 0.7:
            level = 2
        elif val > 0.4:
            level = 1
        else:
            level = 0
            
        one_hot = torch.zeros(1, 3, device=self.device)
        one_hot[0, level] = 1.0
        
        return one_hot, level
    
    def postprocess_output(self, output_tensor, original_size):
        """Postprocess model output to image."""
        output_tensor = self.denormalize(output_tensor.squeeze(0))
        output_tensor = torch.clamp(output_tensor, 0, 1)
        
        output_np = output_tensor.cpu().numpy()
        output_np = (output_np * 255).astype(np.uint8)
        output_np = np.transpose(output_np, (1, 2, 0))
        
        image = Image.fromarray(output_np)
        image = image.resize(original_size, Image.LANCZOS)
        return image
    
    @torch.no_grad()
    def dehaze_image(self, image_bytes):
        """
        Perform image dehazing with high-resolution (Native) support.
        """
        # Load main image
        image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        original_size = image_pil.size
        w, h = original_size
        
        # Estimate haze level (using 512px internally)
        condition, haze_level = self.estimate_haze_level(image_pil)
        
        # Resize to nearest multiple of 32 for U-Net compatibility
        # This keeps the image sharp (no low-res blur)
        new_w = w - (w % 32)
        new_h = h - (h % 32)
        if new_w == 0 or new_h == 0:
            new_w, new_h = 512, 512 # Fallback
            
        img_resized = image_pil.resize((new_w, new_h), Image.LANCZOS)
        
        # Normalize and Move to GPU
        input_tensor = self.normalize(img_resized).unsqueeze(0).to(self.device)
        
        # Run inference
        output_tensor = self.model(input_tensor, condition)
        
        # Postprocess (resizes back to original sharp dimensions)
        dehazed_image = self.postprocess_output(output_tensor, original_size)
        
        return dehazed_image, int(haze_level)


# Global inference engine
inference_engine = None

def initialize_inference(model_path):
    global inference_engine
    inference_engine = DehazingInference(model_path)

def get_inference_engine():
    if inference_engine is None:
        raise RuntimeError("Inference engine not initialized!")
    return inference_engine
