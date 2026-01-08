"""
Test CUDA availability and system setup.

Run this script to verify your system is ready for training and inference.
"""

import sys


def test_cuda():
    """Test CUDA availability."""
    print("="*60)
    print("Testing CUDA Setup")
    print("="*60)
    
    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not installed!")
        print("  Install with: pip install torch torchvision")
        return False
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"{'✓' if cuda_available else '✗'} CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU count: {torch.cuda.device_count()}")
        print(f"✓ GPU name: {torch.cuda.get_device_name(0)}")
        
        # Get GPU memory
        gpu_props = torch.cuda.get_device_properties(0)
        total_memory = gpu_props.total_memory / 1024**3
        print(f"✓ GPU memory: {total_memory:.2f} GB")
        
        # Test tensor creation on GPU
        try:
            test_tensor = torch.randn(100, 100).cuda()
            print("✓ GPU tensor creation successful")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"✗ GPU tensor creation failed: {e}")
            return False
    else:
        print("\n⚠ WARNING: CUDA is not available!")
        print("This project requires CUDA. Please:")
        print("1. Ensure you have an NVIDIA GPU")
        print("2. Install NVIDIA GPU drivers")
        print("3. Install CUDA Toolkit")
        print("4. Reinstall PyTorch with CUDA support:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return False
    
    return True


def test_dependencies():
    """Test required dependencies."""
    print("\n" + "="*60)
    print("Testing Dependencies")
    print("="*60)
    
    dependencies = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('tqdm', 'tqdm'),
        ('matplotlib', 'Matplotlib'),
        ('cv2', 'OpenCV'),
    ]
    
    all_installed = True
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} not installed")
            all_installed = False
    
    return all_installed


def test_model_architecture():
    """Test model can be created."""
    print("\n" + "="*60)
    print("Testing Model Architecture")
    print("="*60)
    
    try:
        import torch
        sys.path.append('training')
        from model import get_model
        
        print("Creating model...")
        model = get_model('cuda' if torch.cuda.is_available() else 'cpu')
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Model created successfully")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        if torch.cuda.is_available():
            print("Testing forward pass...")
            dummy_input = torch.randn(1, 3, 512, 512).cuda()
            dummy_condition = torch.zeros(1, 3).cuda()
            dummy_condition[0, 1] = 1.0  # Medium haze
            
            with torch.no_grad():
                output = model(dummy_input, dummy_condition)
            
            print(f"✓ Forward pass successful")
            print(f"  Input shape: {dummy_input.shape}")
            print(f"  Output shape: {output.shape}")
            
            del model, dummy_input, dummy_condition, output
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset():
    """Test dataset structure."""
    print("\n" + "="*60)
    print("Testing Dataset")
    print("="*60)
    
    import os
    
    data_dir = 'data'
    hazy_dir = os.path.join(data_dir, 'hazy')
    clear_dir = os.path.join(data_dir, 'clear')
    
    if not os.path.exists(data_dir):
        print(f"⚠ Dataset directory not found: {data_dir}")
        print("  Please prepare your dataset first (see DATASET_SETUP.md)")
        return False
    
    if not os.path.exists(hazy_dir) or not os.path.exists(clear_dir):
        print(f"⚠ Missing hazy/ or clear/ subdirectories")
        print("  Expected structure:")
        print("    data/")
        print("    ├── hazy/")
        print("    └── clear/")
        return False
    
    hazy_files = [f for f in os.listdir(hazy_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    clear_files = [f for f in os.listdir(clear_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"✓ Dataset directory found")
    print(f"  Hazy images: {len(hazy_files)}")
    print(f"  Clear images: {len(clear_files)}")
    
    if len(hazy_files) == 0 or len(clear_files) == 0:
        print("⚠ No images found in dataset")
        return False
    
    if len(hazy_files) != len(clear_files):
        print("⚠ Mismatch between hazy and clear image counts")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Weather-Adaptive Image Dehazing System")
    print("System Verification Test")
    print("="*60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("CUDA Setup", test_cuda()))
    results.append(("Dependencies", test_dependencies()))
    results.append(("Model Architecture", test_model_architecture()))
    results.append(("Dataset", test_dataset()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:.<40} {status}")
    
    all_passed = all(result[1] for result in results[:3])  # Dataset is optional
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ System is ready for training and inference!")
        if not results[3][1]:
            print("⚠ Note: Dataset not found. Prepare dataset before training.")
    else:
        print("✗ System is NOT ready. Please fix the issues above.")
    print("="*60 + "\n")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
