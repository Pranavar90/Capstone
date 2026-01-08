# System Architecture

This document describes the architecture of the weather-adaptive image dehazing system.

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Web Application                          │
│                      (Next.js)                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Upload     │  │  Processing  │  │  Comparison  │      │
│  │  Component   │→ │    Loader    │→ │    Slider    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↓ HTTP POST
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Backend                            │
│                   (Python + CUDA)                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Inference Engine                        │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐    │   │
│  │  │Preprocess  │→ │   Model    │→ │Postprocess │    │   │
│  │  └────────────┘  └────────────┘  └────────────┘    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Dehazing Model                             │
│                   (PyTorch + CUDA)                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         ResNet-18 Feature Extractor                  │   │
│  │              (Frozen Weights)                        │   │
│  │              ImageNet Pretrained                     │   │
│  └──────────────────────────────────────────────────────┘   │
│                            ↓                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Conditional U-Net Dehazing Network           │   │
│  │                                                      │   │
│  │  Encoder → Bottleneck + Conditioning → Decoder      │   │
│  │              ↑                                       │   │
│  │         3-Class Haze Severity                        │   │
│  │      (Light / Medium / Heavy)                        │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Web Application (Frontend)

**Technology**: Next.js 14 (App Router), TypeScript, Framer Motion

**Components**:
- `ImageUploader`: Drag-and-drop upload with animations
- `ProcessingLoader`: Animated loading state with progress indicators
- `ImageComparison`: Interactive slider for before/after comparison
- `page.tsx`: Main application logic and state management

**Features**:
- Smooth animations and transitions
- Responsive design
- Real-time feedback
- Download functionality

**Flow**:
1. User uploads image
2. Image sent to backend via HTTP POST
3. Processing animation displayed
4. Results shown with comparison slider
5. User can download dehazed image

### 2. FastAPI Backend

**Technology**: FastAPI, Python, PyTorch

**Modules**:
- `main.py`: API endpoints and request handling
- `inference.py`: Model loading and inference logic

**Endpoints**:
- `GET /`: API information
- `GET /health`: Health check
- `POST /dehaze/image`: Image dehazing (functional)
- `POST /dehaze/video`: Video dehazing (placeholder)

**Features**:
- CUDA enforcement
- GPU-based inference
- Automatic preprocessing/postprocessing
- CORS enabled for frontend
- Error handling

**Flow**:
1. Receive image upload
2. Preprocess (resize, normalize)
3. Estimate haze severity
4. Run model inference on GPU
5. Postprocess (denormalize, resize)
6. Return dehazed image with metadata

### 3. Dehazing Model

**Technology**: PyTorch, CUDA

**Architecture**:

#### ResNet-18 Feature Extractor
- Pretrained on ImageNet
- All weights frozen (no training)
- Used for feature extraction
- Provides rich visual features

#### Conditional U-Net
- **Encoder**: 4 downsampling blocks
  - Conv → BatchNorm → ReLU
  - MaxPooling for downsampling
  - Channels: 64 → 128 → 256 → 512

- **Bottleneck**: Feature fusion
  - Concatenate encoder features with conditioning
  - Conditioning: 3-class one-hot vector
  - Spatially replicated to match feature map size

- **Decoder**: 4 upsampling blocks
  - Transposed convolution for upsampling
  - Skip connections from encoder
  - Channels: 512 → 256 → 128 → 64

- **Output**: 1×1 convolution to 3 channels (RGB)

**Conditioning Strategy**:
- 3 haze severity classes: Light (0), Medium (1), Heavy (2)
- One-hot encoded: [1,0,0], [0,1,0], or [0,0,1]
- Injected at bottleneck via concatenation
- Allows model to adapt to different haze levels

**Future Improvement**:
- K-means clustering for automatic haze classification
- Multi-scale feature fusion
- Attention mechanisms

### 4. Training Pipeline

**Technology**: PyTorch, CUDA

**Modules**:
- `train.py`: Main training loop
- `model.py`: Model architecture
- `dataset.py`: Data loading and preprocessing
- `create_synthetic_dataset.py`: Synthetic data generation

**Training Configuration**:
- **Loss**: L1 (Mean Absolute Error)
- **Optimizer**: Adam (lr=1e-4)
- **Batch Size**: 2-4 (RTX 3050 optimized)
- **Epochs**: ≤10
- **Image Size**: 512×512
- **Normalization**: ImageNet mean/std

**Data Augmentation**:
- Minimal (resize only)
- No random cropping or heavy augmentation
- Focus on paired alignment

**Output**:
- Best model: `models/dehaze_model_best.pth`
- Final model: `models/dehaze_model_final.pth`
- Training curves: `models/training_curves.png`

## Data Flow

### Training Flow

```
Dataset (Kaggle)
    ↓
data/hazy/ + data/clear/
    ↓
DataLoader (batch_size=4)
    ↓
Preprocessing (resize, normalize)
    ↓
Model (forward pass)
    ↓
L1 Loss (vs. ground truth)
    ↓
Backpropagation
    ↓
Adam Optimizer (update weights)
    ↓
Save best model
```

### Inference Flow

```
User uploads image
    ↓
Frontend → Backend API
    ↓
Preprocess:
  - Resize to 512×512
  - Convert to tensor
  - Normalize (ImageNet)
    ↓
Estimate haze level:
  - Calculate mean brightness
  - Classify: light/medium/heavy
  - One-hot encode
    ↓
Model inference (GPU):
  - ResNet-18 features (optional)
  - U-Net with conditioning
  - Output: dehazed tensor
    ↓
Postprocess:
  - Denormalize
  - Clamp [0, 1]
  - Resize to original size
  - Convert to image
    ↓
Return to frontend
    ↓
Display with comparison slider
```

## Technology Stack Summary

### Machine Learning
- **Framework**: PyTorch 2.0+
- **Acceleration**: CUDA 11.0+
- **Pretrained Model**: ResNet-18 (ImageNet)
- **Custom Model**: Conditional U-Net

### Backend
- **Framework**: FastAPI
- **Language**: Python 3.8+
- **Server**: Uvicorn

### Frontend
- **Framework**: Next.js 14
- **Language**: TypeScript
- **Styling**: CSS + Tailwind
- **Animations**: Framer Motion
- **Icons**: Lucide React

### Development
- **Package Manager**: NPM (frontend), pip (backend)
- **Version Control**: Git
- **Dataset Source**: Kaggle

## Performance Characteristics

### Training (RTX 3050, 1000 images)
- Batch size: 4
- Images/second: ~8-12
- Epoch time: ~5-10 minutes
- Total time (10 epochs): ~1-2 hours

### Inference (RTX 3050)
- Single image (512×512): ~0.5-1.5 seconds
- Includes preprocessing and postprocessing
- GPU memory usage: ~2-3 GB

### Web Application
- Initial load: <2 seconds
- Upload to result: ~2-4 seconds (including network)
- Smooth 60 FPS animations

## Scalability Considerations

### Current Limitations
- Single image processing only
- No batch processing
- No video support
- Fixed image size (512×512)

### Future Improvements
1. **Batch Processing**: Process multiple images simultaneously
2. **Video Support**: Frame-by-frame with temporal consistency
3. **Dynamic Sizing**: Support arbitrary image sizes
4. **Model Optimization**: TensorRT, ONNX, quantization
5. **Cloud Deployment**: Docker, Kubernetes, serverless
6. **Caching**: Redis for frequently processed images
7. **Queue System**: Celery for async processing

## Security Considerations

### Current Implementation
- CORS enabled (all origins)
- No authentication
- No rate limiting
- No input validation (beyond file type)

### Production Recommendations
1. **Authentication**: JWT tokens, API keys
2. **Rate Limiting**: Prevent abuse
3. **Input Validation**: File size, type, content
4. **CORS**: Restrict to specific origins
5. **HTTPS**: Encrypt data in transit
6. **Monitoring**: Log requests, errors, performance

## Deployment Architecture

### Development
```
localhost:3000 (Frontend)
    ↓
localhost:8000 (Backend)
    ↓
Local GPU (CUDA)
```

### Production (Example)
```
Vercel/Netlify (Frontend)
    ↓
AWS EC2 / GCP Compute (Backend + GPU)
    ↓
S3 / Cloud Storage (Models, datasets)
    ↓
CloudWatch / Stackdriver (Monitoring)
```

## Transparency & Attribution

### Pretrained Components
- **ResNet-18**: Torchvision, pretrained on ImageNet
- **Normalization**: ImageNet mean/std values

### Dataset
- Source: Kaggle (user-provided)
- Type: Paired hazy/clear images
- License: Varies by dataset

### Acceleration
- **GPU**: NVIDIA RTX 3050 (or better)
- **CUDA**: Required for all ML operations

### Future Extensions
- Video dehazing (planned)
- K-means haze classification (planned)
- Real-time processing (planned)

## References

### Papers
- U-Net: Convolutional Networks for Biomedical Image Segmentation
- Deep Residual Learning for Image Recognition (ResNet)
- Atmospheric Scattering Model for Image Dehazing

### Datasets
- RESIDE: A Benchmark for Single Image Dehazing
- O-HAZE: Outdoor Haze Dataset
- I-HAZE: Indoor Haze Dataset

### Technologies
- PyTorch: https://pytorch.org
- FastAPI: https://fastapi.tiangolo.com
- Next.js: https://nextjs.org
- Framer Motion: https://www.framer.com/motion
