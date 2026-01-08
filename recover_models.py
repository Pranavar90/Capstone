import os
import shutil

# Create local models directory
os.makedirs('models', exist_ok=True)

# List of files to recover
files = ['dehaze_model_best.pth', 'dehaze_model_final.pth', 'training_curves.png']

# Check parent directory (../models) relative to CWD
src_dir = '../models'

print(f"Checking for trained models in {os.path.abspath(src_dir)}...")

for f in files:
    src_path = os.path.join(src_dir, f)
    dst_path = os.path.join('models', f)
    
    if os.path.exists(src_path):
        print(f"Copying {f} to local models/ directory...")
        try:
            shutil.copy2(src_path, dst_path)
            print("✓ Success")
        except Exception as e:
            print(f"✗ Failed to copy: {e}")
    else:
        print(f"⚠ File not found: {src_path}")

print("\nModel recovery complete.")
