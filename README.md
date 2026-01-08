# Weather-Adaptive Image Dehazing System

A complete end-to-end image dehazing system with deep learning and modern web interface.

## 🎯 System Overview

This project implements a **weather-adaptive image dehazing system** consisting of:
- PyTorch-based training pipeline (CUDA-accelerated)
- FastAPI inference backend
- Next.js animated web application

## 🔧 Technical Stack

### Machine Learning
- **Framework**: PyTorch (CUDA-only, tested on RTX 3050)
- **Feature Extractor**: ResNet-18 (ImageNet pretrained, frozen weights)
- **Dehazing Network**: Conditional U-Net with 3-class haze severity conditioning
- **Dataset**: Kaggle paired image dehazing dataset (hazy → clear)
- **Training**: L1 loss, Adam optimizer, 10 epochs max

### Backend
- **Framework**: FastAPI
- **Acceleration**: CUDA (GPU inference)
- **Endpoints**: 
  - `/dehaze/image` - Fully functional image dehazing
  - `/dehaze/video` - Placeholder (future extension)

### Frontend
- **Framework**: Next.js (App Router, NPM-based)
- **Animations**: Framer Motion
- **Features**: Image upload, comparison view, download results

## 📋 Requirements

### Hardware
- NVIDIA RTX 3050 or better
- CUDA-capable GPU (required)

### Software
- Python 3.8+
- Node.js 18+
- CUDA Toolkit
- NPM

## 🚀 Quick Start

### 1. Dataset Setup

Download a paired dehazing dataset from Kaggle:

```bash
# Install Kaggle API
pip install kaggle

# Configure Kaggle credentials (place kaggle.json in ~/.kaggle/)
# Download dataset (example - replace with actual dataset)
kaggle datasets download -d [dataset-name]
unzip [dataset-name].zip -d data/
```

Expected structure:
```
data/
├── hazy/
│   ├── image1.jpg
│   └── ...
└── clear/
    ├── image1.jpg
    └── ...
```

### 2. Training

```bash
cd training
pip install -r requirements.txt
python train.py --data_dir ../data --epochs 10 --batch_size 4
```

### 3. Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Frontend

```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:3000`

## 🏗️ Architecture

### Model Architecture
```
Input Image (512×512)
    ↓
ResNet-18 Feature Extractor (frozen)
    ↓
Conditional U-Net Dehazing Network
    ├── Encoder (downsampling)
    ├── Bottleneck + Conditioning Injection
    └── Decoder (upsampling)
    ↓
Dehazed Output (512×512)
```

### Conditioning Strategy
- 3-class haze severity: Light, Medium, Heavy
- One-hot encoded conditioning vector
- Injected at bottleneck via concatenation

### Future Improvements
- K-means clustering for automatic haze severity classification
- Video dehazing support
- Real-time processing optimization

## 📊 Training Details

- **Image Size**: 512×512
- **Normalization**: ImageNet mean/std
- **Loss**: L1 (Mean Absolute Error)
- **Optimizer**: Adam
- **Batch Size**: 2-4 (RTX 3050 optimized)
- **Epochs**: ≤10
- **Augmentation**: Minimal (resize only)

## 🌐 Web Interface Features

- **Upload**: Drag-and-drop or click to upload
- **Processing**: Animated loading states
- **Results**: Side-by-side comparison with slider
- **Download**: Export dehazed images
- **Animations**: Smooth transitions using Framer Motion

## 📝 Transparency & Attribution

- **ResNet-18**: Pretrained on ImageNet (torchvision)
- **CUDA Acceleration**: Optimized for NVIDIA RTX 3050
- **Dataset**: Sourced from Kaggle (paired hazy/clear images)
- **Video Dehazing**: Planned future extension

## 📂 Project Structure

```
capstonr/
├── training/
│   ├── train.py
│   ├── model.py
│   ├── dataset.py
│   └── requirements.txt
├── backend/
│   ├── main.py
│   ├── inference.py
│   └── requirements.txt
├── frontend/
│   ├── app/
│   ├── components/
│   ├── public/
│   └── package.json
├── data/
│   ├── hazy/
│   └── clear/
└── models/
    └── dehaze_model.pth
```

## 🎓 Academic Context

This system demonstrates:
- Transfer learning with frozen feature extractors
- Conditional image-to-image translation
- End-to-end ML deployment
- Modern web UI/UX practices

## 📄 License

MIT License - See LICENSE file for details
