"""
Rename dataset files to have matching names.
"""
import os
from pathlib import Path

# Rename hazy images
hazy_dir = Path('data/hazy')
for img in sorted(hazy_dir.glob('*_hazy.jpg')):
    number = img.stem.split('_')[0]
    new_name = f'image_{number}.jpg'
    img.rename(hazy_dir / new_name)
    print(f'Renamed {img.name} -> {new_name}')

# Rename clear images
clear_dir = Path('data/clear')
for img in sorted(clear_dir.glob('*_GT.jpg')):
    number = img.stem.split('_')[0]
    new_name = f'image_{number}.jpg'
    img.rename(clear_dir / new_name)
    print(f'Renamed {img.name} -> {new_name}')

print('\nDataset organization complete!')
print(f'Hazy images: {len(list(hazy_dir.glob("*.jpg")))}')
print(f'Clear images: {len(list(clear_dir.glob("*.jpg")))}')
