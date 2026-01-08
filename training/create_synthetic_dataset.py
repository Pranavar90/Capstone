"""
Create a synthetic hazy dataset for testing.

This script generates synthetic hazy images from clear images
using atmospheric scattering model.

Usage:
    python create_synthetic_dataset.py --input_dir <clear_images> --output_dir ../data --num_samples 100
"""

import os
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def add_haze(image, beta=0.8, A=0.85, depth_type='gradient'):
    """
    Add synthetic haze to an image using atmospheric scattering model.
    
    Haze model: I(x) = J(x) * t(x) + A * (1 - t(x))
    where:
        I(x) = observed hazy image
        J(x) = scene radiance (clear image)
        t(x) = transmission map = exp(-beta * d(x))
        d(x) = depth map
        A = atmospheric light
        beta = scattering coefficient
    
    Args:
        image: Input clear image (BGR)
        beta: Scattering coefficient (higher = more haze)
        A: Atmospheric light intensity (0-1)
        depth_type: Type of depth map ('gradient', 'random', 'center')
    
    Returns:
        Hazy image (BGR)
    """
    height, width = image.shape[:2]
    
    # Create depth map
    if depth_type == 'gradient':
        # Horizontal gradient (haze increases with distance)
        depth = np.linspace(0, 1, width)
        depth = np.tile(depth, (height, 1))
    elif depth_type == 'random':
        # Random depth (patchy haze)
        depth = np.random.rand(height, width) * 0.8 + 0.2
    elif depth_type == 'center':
        # Radial gradient from center
        y, x = np.ogrid[:height, :width]
        cx, cy = width // 2, height // 2
        depth = np.sqrt((x - cx)**2 + (y - cy)**2)
        depth = depth / depth.max()
    else:
        depth = np.ones((height, width)) * 0.5
    
    # Compute transmission map
    transmission = np.exp(-beta * depth)
    transmission = np.stack([transmission] * 3, axis=2)
    
    # Apply haze model
    image_normalized = image.astype(np.float32) / 255.0
    hazy = image_normalized * transmission + A * (1 - transmission)
    hazy = np.clip(hazy * 255, 0, 255).astype(np.uint8)
    
    return hazy


def create_synthetic_dataset(input_dir, output_dir, num_samples=None):
    """
    Create synthetic hazy dataset from clear images.
    
    Args:
        input_dir: Directory containing clear images
        output_dir: Output directory (will create hazy/ and clear/ subdirs)
        num_samples: Number of samples to generate (None = all)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    hazy_dir = output_path / 'hazy'
    clear_dir = output_path / 'clear'
    hazy_dir.mkdir(parents=True, exist_ok=True)
    clear_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of input images
    image_files = list(input_path.glob('*.jpg')) + \
                  list(input_path.glob('*.jpeg')) + \
                  list(input_path.glob('*.png'))
    
    if num_samples:
        image_files = image_files[:num_samples]
    
    print(f"Processing {len(image_files)} images...")
    
    # Process each image
    for i, img_path in enumerate(tqdm(image_files)):
        try:
            # Read clear image
            clear = cv2.imread(str(img_path))
            if clear is None:
                print(f"Warning: Could not read {img_path}")
                continue
            
            # Generate hazy version with denser random parameters
            beta = np.random.uniform(1.0, 2.5)  # Increased haze density (High Clarity)
            A = np.random.uniform(0.8, 1.0)     # Brighter atmospheric light
            depth_type = np.random.choice(['gradient', 'random', 'center'])
            
            hazy = add_haze(clear, beta=beta, A=A, depth_type=depth_type)
            
            # Save both images with consistent naming
            output_name = f'image_{i:04d}.jpg'
            cv2.imwrite(str(hazy_dir / output_name), hazy)
            cv2.imwrite(str(clear_dir / output_name), clear)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print(f"\nDataset created successfully!")
    print(f"Hazy images: {len(list(hazy_dir.glob('*.jpg')))}")
    print(f"Clear images: {len(list(clear_dir.glob('*.jpg')))}")
    print(f"\nDataset location: {output_path}")


def download_sample_images(output_dir, num_images=100):
    """
    Download sample images for testing (placeholder).
    
    In practice, you would:
    1. Use a dataset from Kaggle
    2. Use your own images
    3. Download from a public dataset
    
    This is just a placeholder to show the structure.
    """
    print("Note: Automatic download not implemented.")
    print("Please provide your own clear images or download from Kaggle.")
    print("See DATASET_SETUP.md for instructions.")


def main():
    parser = argparse.ArgumentParser(
        description='Create synthetic hazy dataset for testing'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing clear images'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../data',
        help='Output directory (default: ../data)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Number of samples to generate (default: all)'
    )
    
    args = parser.parse_args()
    
    # Verify input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist!")
        print("\nTo get started:")
        print("1. Download a dataset from Kaggle (see DATASET_SETUP.md)")
        print("2. Or provide your own clear images")
        return
    
    # Create synthetic dataset
    create_synthetic_dataset(
        args.input_dir,
        args.output_dir,
        args.num_samples
    )


if __name__ == '__main__':
    main()
