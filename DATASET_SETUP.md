# Dataset Setup Guide

This guide will help you download and prepare a paired image dehazing dataset from Kaggle.

## Prerequisites

1. **Kaggle Account**: Create a free account at [kaggle.com](https://www.kaggle.com)
2. **Kaggle API Token**: 
   - Go to your Kaggle account settings
   - Click "Create New API Token"
   - This downloads `kaggle.json`
   - Place it in `~/.kaggle/` (Linux/Mac) or `C:\Users\<YourUsername>\.kaggle\` (Windows)

## Recommended Datasets

Here are some good paired dehazing datasets available on Kaggle:

### Option 1: Dense-Haze Dataset
```bash
kaggle datasets download -d rajat95gupta/hazing-images-dataset
unzip hazing-images-dataset.zip -d data/
```

### Option 2: O-HAZE Dataset
```bash
kaggle datasets download -d balraj98/outdoor-haze-dataset-ohaze
unzip outdoor-haze-dataset-ohaze.zip -d data/
```

### Option 3: I-HAZE Dataset
```bash
kaggle datasets download -d balraj98/indoor-haze-dataset-ihaze
unzip indoor-haze-dataset-ihaze.zip -d data/
```

### Option 4: Custom Search
Search Kaggle for "image dehazing" or "hazy images" and choose a dataset with:
- Paired hazy/clear images
- At least 100+ image pairs
- Simple folder structure

## Dataset Structure

After downloading, organize your data as follows:

```
capstonr/
└── data/
    ├── hazy/
    │   ├── image_001.jpg
    │   ├── image_002.jpg
    │   └── ...
    └── clear/
        ├── image_001.jpg
        ├── image_002.jpg
        └── ...
```

**Important**: Ensure that:
- Hazy and clear images have matching filenames
- Images are in common formats (JPG, PNG, JPEG)
- Both folders contain the same number of images

## Manual Organization

If your downloaded dataset has a different structure, you may need to reorganize it:

```python
# Example reorganization script
import os
import shutil
from pathlib import Path

# Adjust these paths based on your downloaded dataset
source_hazy = "data/downloaded/hazy_images"
source_clear = "data/downloaded/clear_images"

dest_hazy = "data/hazy"
dest_clear = "data/clear"

# Create destination directories
os.makedirs(dest_hazy, exist_ok=True)
os.makedirs(dest_clear, exist_ok=True)

# Copy and rename files
for i, (hazy_file, clear_file) in enumerate(zip(
    sorted(Path(source_hazy).glob("*.jpg")),
    sorted(Path(source_clear).glob("*.jpg"))
)):
    shutil.copy(hazy_file, f"{dest_hazy}/image_{i:04d}.jpg")
    shutil.copy(clear_file, f"{dest_clear}/image_{i:04d}.jpg")
```

## Verify Dataset

Before training, verify your dataset:

```python
import os

hazy_dir = "data/hazy"
clear_dir = "data/clear"

hazy_files = set(os.listdir(hazy_dir))
clear_files = set(os.listdir(clear_dir))

print(f"Hazy images: {len(hazy_files)}")
print(f"Clear images: {len(clear_files)}")
print(f"Matching pairs: {len(hazy_files & clear_files)}")

if hazy_files != clear_files:
    print("\nWarning: Mismatch detected!")
    print(f"Only in hazy: {hazy_files - clear_files}")
    print(f"Only in clear: {clear_files - hazy_files}")
else:
    print("\n✓ Dataset verified successfully!")
```

## Dataset Size Recommendations

For this project:
- **Minimum**: 100 image pairs (for quick testing)
- **Recommended**: 500-1000 image pairs (for decent results)
- **Optimal**: 2000+ image pairs (for best performance)

Note: With RTX 3050 and batch size 4, training on 1000 images takes approximately:
- 1 epoch: ~5-10 minutes
- 10 epochs: ~1-2 hours

## Alternative: Create Synthetic Dataset

If you can't access Kaggle, you can create a synthetic hazy dataset:

```python
import cv2
import numpy as np
from pathlib import Path

def add_haze(image, beta=0.8, A=0.85):
    """Add synthetic haze to an image."""
    height, width = image.shape[:2]
    
    # Create depth map (simple gradient)
    depth = np.linspace(0, 1, width)
    depth = np.tile(depth, (height, 1))
    
    # Apply haze model: I = J * t + A * (1 - t)
    # where t = exp(-beta * depth)
    transmission = np.exp(-beta * depth)
    transmission = np.stack([transmission] * 3, axis=2)
    
    hazy = image * transmission + A * 255 * (1 - transmission)
    return np.clip(hazy, 0, 255).astype(np.uint8)

# Process clear images to create hazy versions
clear_dir = Path("data/clear_images")  # Your clear images
hazy_dir = Path("data/hazy")
clear_dest = Path("data/clear")

hazy_dir.mkdir(parents=True, exist_ok=True)
clear_dest.mkdir(parents=True, exist_ok=True)

for img_path in clear_dir.glob("*.jpg"):
    # Read clear image
    clear = cv2.imread(str(img_path))
    
    # Generate hazy version
    hazy = add_haze(clear, beta=np.random.uniform(0.5, 1.2))
    
    # Save both
    cv2.imwrite(str(hazy_dir / img_path.name), hazy)
    cv2.imwrite(str(clear_dest / img_path.name), clear)

print("Synthetic dataset created!")
```

## Next Steps

Once your dataset is ready:

1. Verify the structure matches the expected format
2. Run the training script:
   ```bash
   cd training
   python train.py --data_dir ../data --epochs 10 --batch_size 4
   ```

3. Monitor training progress and wait for completion

4. The trained model will be saved in `models/dehaze_model_best.pth`
