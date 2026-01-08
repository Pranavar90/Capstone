# Quick Start Guide

This guide will help you get the weather-adaptive image dehazing system up and running.

## System Requirements

### Hardware
- **GPU**: NVIDIA RTX 3050 or better
- **VRAM**: 4GB+ recommended
- **RAM**: 8GB+ recommended
- **Storage**: 5GB+ free space

### Software
- **OS**: Windows (tested), Linux, or macOS
- **Python**: 3.8 or higher
- **Node.js**: 18 or higher
- **CUDA**: 11.0 or higher
- **NPM**: Latest version

## Installation Steps

### 1. Verify CUDA Installation

```bash
# Check if CUDA is available
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

If CUDA is not available, install it from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads).

### 2. Setup Training Environment

```bash
cd training
pip install -r requirements.txt
```

### 3. Prepare Dataset

Follow the instructions in `DATASET_SETUP.md` to download and prepare your dataset.

Quick option for testing:
```bash
# Create a small test dataset (synthetic)
python -c "
import os
os.makedirs('data/hazy', exist_ok=True)
os.makedirs('data/clear', exist_ok=True)
print('Test directories created. Add some images to get started.')
"
```

### 4. Train the Model

```bash
cd training
python train.py --data_dir ../data --epochs 10 --batch_size 4
```

**Expected training time** (RTX 3050, 1000 images):
- Per epoch: ~5-10 minutes
- Total (10 epochs): ~1-2 hours

**Training output:**
- Model weights: `models/dehaze_model_best.pth`
- Training curves: `models/training_curves.png`

### 5. Setup Backend

```bash
cd backend
pip install -r requirements.txt
```

### 6. Start Backend Server

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at `http://localhost:8000`

**Verify backend:**
- Open `http://localhost:8000` in your browser
- You should see API information
- Check `http://localhost:8000/health` for health status

### 7. Setup Frontend

```bash
cd frontend
npm install
```

### 8. Start Frontend

```bash
cd frontend
npm run dev
```

The web application will be available at `http://localhost:3000`

## Usage

### Image Dehazing

1. Open `http://localhost:3000` in your browser
2. Upload a hazy image (drag-and-drop or click to browse)
3. Wait for processing (usually 1-3 seconds)
4. View the comparison slider and side-by-side results
5. Download the dehazed image

### Video Dehazing (Coming Soon)

Video dehazing is planned for a future release. The UI includes a placeholder for this feature.

## Troubleshooting

### CUDA Not Available

**Problem**: `CUDA is not available!` error

**Solutions**:
1. Install NVIDIA GPU drivers
2. Install CUDA Toolkit from NVIDIA
3. Reinstall PyTorch with CUDA support:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### Model Not Found

**Problem**: `Model not found` error when starting backend

**Solution**: Train the model first:
```bash
cd training
python train.py --data_dir ../data --epochs 10 --batch_size 4
```

### Out of Memory Error

**Problem**: `CUDA out of memory` during training

**Solutions**:
1. Reduce batch size:
   ```bash
   python train.py --batch_size 2
   ```
2. Close other GPU-intensive applications
3. Reduce image size (edit `dataset.py`)

### Backend Connection Error

**Problem**: Frontend can't connect to backend

**Solutions**:
1. Ensure backend is running on port 8000
2. Check firewall settings
3. Verify CORS is enabled in `backend/main.py`

### Slow Inference

**Problem**: Image processing takes too long

**Solutions**:
1. Ensure model is on GPU (check backend logs)
2. Close other GPU applications
3. Use smaller images

## Performance Benchmarks

### RTX 3050 (4GB VRAM)

**Training:**
- Batch size: 4
- Images/second: ~8-12
- Epoch time (1000 images): ~5-10 minutes

**Inference:**
- Single image (512×512): ~0.5-1.5 seconds
- Includes preprocessing and postprocessing

### RTX 3060 (12GB VRAM)

**Training:**
- Batch size: 8
- Images/second: ~15-20
- Epoch time (1000 images): ~3-5 minutes

**Inference:**
- Single image (512×512): ~0.3-0.8 seconds

## API Endpoints

### GET /
Returns API information and system status

### GET /health
Health check endpoint
```json
{
  "status": "healthy",
  "cuda_available": true,
  "gpu": "NVIDIA GeForce RTX 3050",
  "model_loaded": true
}
```

### POST /dehaze/image
Dehaze a single image

**Request:**
- Content-Type: `multipart/form-data`
- Body: Image file

**Response:**
- Content-Type: `image/jpeg`
- Headers:
  - `X-Haze-Level`: Detected haze severity (light/medium/heavy)
  - `X-Haze-Level-Index`: Numeric index (0/1/2)

### POST /dehaze/video
Placeholder for video dehazing (returns 501 Not Implemented)

## Development

### Project Structure

```
capstonr/
├── training/           # PyTorch training pipeline
│   ├── train.py       # Main training script
│   ├── model.py       # Model architecture
│   ├── dataset.py     # Dataset loader
│   └── requirements.txt
├── backend/           # FastAPI inference server
│   ├── main.py       # API endpoints
│   ├── inference.py  # Inference engine
│   └── requirements.txt
├── frontend/          # Next.js web application
│   ├── app/          # App router pages
│   ├── components/   # React components
│   └── package.json
├── data/             # Dataset (not in repo)
│   ├── hazy/
│   └── clear/
└── models/           # Trained models (not in repo)
    └── dehaze_model_best.pth
```

### Making Changes

**Model architecture:**
- Edit `training/model.py`
- Retrain the model
- Restart backend

**Backend API:**
- Edit `backend/main.py` or `backend/inference.py`
- Backend auto-reloads with `--reload` flag

**Frontend UI:**
- Edit files in `frontend/app/` or `frontend/components/`
- Frontend auto-reloads in dev mode

## Next Steps

1. **Improve model**: Train on larger dataset, tune hyperparameters
2. **Add features**: Batch processing, video support, custom conditioning
3. **Deploy**: Use Docker, cloud platforms (AWS, GCP, Azure)
4. **Optimize**: Model quantization, TensorRT, ONNX export

## Support

For issues or questions:
1. Check this guide and README.md
2. Review error messages carefully
3. Verify all requirements are met
4. Check GPU and CUDA availability

## License

MIT License - See LICENSE file for details
