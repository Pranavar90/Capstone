"""
Dataset loader for paired hazy/clear images.
Expects folder structure:
    data/
    ├── hazy/
    │   ├── image1.jpg
    │   └── ...
    └── clear/
        ├── image1.jpg
        └── ...
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.signal import find_peaks
from sklearn.cluster import KMeans


class DehazingDataset(Dataset):
    """
    Paired image dataset for dehazing.
    Loads hazy and corresponding clear images.
    """
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Root directory containing 'hazy' and 'clear' subdirectories
            transform: Optional transform to apply to images
        """
        self.hazy_dir = os.path.join(data_dir, 'hazy')
        self.clear_dir = os.path.join(data_dir, 'clear')
        
        # Get list of image files
        self.image_files = sorted([f for f in os.listdir(self.hazy_dir) 
                                   if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        # Default transform: resize to 512x512 (User Requested)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        # Pre-calculate haze levels for speed (NEW)
        print("Pre-calculating haze levels...")
        self.haze_levels = []
        for img_name in self.image_files:
            p = os.path.join(self.hazy_dir, img_name)
            img = Image.open(p).convert('RGB')
            self.haze_levels.append(self._estimate_haze_level(img))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            hazy_tensor: Preprocessed hazy image tensor
            clear_tensor: Preprocessed clear image tensor
            haze_level: Pre-calculated one-hot encoded haze severity
        """
        img_name = self.image_files[idx]
        haze_level = self.haze_levels[idx]
        
        # Load images
        hazy_path = os.path.join(self.hazy_dir, img_name)
        clear_path = os.path.join(self.clear_dir, img_name)
        
        hazy_img = Image.open(hazy_path).convert('RGB')
        clear_img = Image.open(clear_path).convert('RGB')
        
        # Apply transforms
        hazy_tensor = self.transform(hazy_img)
        clear_tensor = self.transform(clear_img)
        
        return hazy_tensor, clear_tensor, haze_level
    
    def _estimate_haze_level(self, image_pil):
        """
        Estimate haze severity using Histogram Peaks + K-Means (User Logic).
        
        Logic:
        1. Convert to Grayscale.
        2. Compute Histogram (256 bins).
        3. Identify Peaks (using find_peaks).
        4. Set num_clusters = num_peaks.
        5. Run K-Means Clustering on pixel intensities.
        6. Determine severity based on cluster centers.
        
        Returns:
            One-hot encoded tensor of shape (3,)
        """
        # Convert to numpy array (RGB -> Gray)
        img_np = np.array(image_pil)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # 1. Histogram & Peaks
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        
        # Smooth histogram lightly to avoid noise peaks
        hist_smooth = np.convolve(hist, np.ones(5)/5, mode='same')
        
        # Find peaks
        peaks, _ = find_peaks(hist_smooth, distance=20, prominence=10)
        num_peaks = len(peaks)
        
        # Constraint: num_clusters = num_peaks
        n_clusters = num_peaks
        if n_clusters < 1: n_clusters = 1  # Fallback
        
        # 2. K-Means
        # Reshape for clustering
        pixels = gray.reshape(-1, 1)
        
        # Subsample for performance (5000 pixels is enough for distribution)
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
            # Fallback if clustering fails
            centers = [np.mean(pixels)]
            
        # 3. Classify Severity
        # Haze generally increases brightness (atmospheric light) and reduces contrast (clusters clumped?)
        # Simple heuristic mapping from cluster centers:
        # High brightness usually indicates thick haze (in daytime).
        # We use the max center or mean center.
        
        val = centers[-1] / 255.0  # Normalized brightest center
        
        if val > 0.7:
            level = 2  # Heavy
        elif val > 0.4:
            level = 1  # Medium
        else:
            level = 0  # Light
            
        # One-hot encode
        one_hot = torch.zeros(3)
        one_hot[level] = 1.0
        
        return one_hot


def get_dataloaders(data_dir, batch_size=4, num_workers=4):
    """
    Create train and validation dataloaders.
    """
    dataset = DehazingDataset(data_dir)
    
    # Split 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader
