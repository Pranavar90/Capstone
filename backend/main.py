"""
FastAPI backend for image dehazing inference.

Endpoints:
    - POST /dehaze/image: Dehaze a single image (fully functional)
    - POST /dehaze/video: Placeholder for video dehazing (future extension)
    - GET /health: Health check endpoint

Requirements:
    - CUDA-capable GPU
    - Trained model weights in ../models/
"""

import sys
import os
import io
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch

# Add current directory to path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import initialize_inference, get_inference_engine


# Create FastAPI app
app = FastAPI(
    title="Weather-Adaptive Image Dehazing API",
    description="CUDA-accelerated image dehazing using conditional U-Net",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """
    Initialize inference engine on startup.
    Loads model onto GPU and keeps it there for fast inference.
    """
    # Enforce CUDA requirement
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available! This application requires a CUDA-capable GPU."
        )
    
    # Get absolute path to models directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    
    print("="*60)
    print("Starting Research Computing Backend: Dehazing Node")
    print("="*60)
    print(f"CUDA INFRASTRUCTURE: {torch.cuda.get_device_name(0)}")
    print(f"COMPUTE CAPABILITY:  {torch.version.cuda}")
    print(f"MODEL REPOSITORY:    {models_dir}")
    print("="*60)
    
    # Load model
    model_path = os.path.join(models_dir, 'dehaze_model_best.pth')
    
    if not os.path.exists(model_path):
        print(f"WARNING: Preferred weights not found at {model_path}")
        model_path = os.path.join(models_dir, 'dehaze_model_final.pth')
    
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found in {models_dir}")
            
        initialize_inference(model_path)
        print("✓ ACTIVE: Neural weights synchronized")
        print("="*60)
    except Exception as e:
        print(f"ERROR: Weight synchronization failed: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Weather-Adaptive Image Dehazing API",
        "version": "1.0.0",
        "description": "CUDA-accelerated image dehazing",
        "endpoints": {
            "/dehaze": "POST - Dehaze a single image",
            "/dehaze/video": "POST - Video dehazing (Work in Progress)",
            "/health": "GET - Health check"
        },
        "technology": {
            "framework": "PyTorch",
            "model": "Conditional U-Net with ResNet-18 feature extractor",
            "pretrained": "ImageNet (ResNet-18)",
            "acceleration": "CUDA (GPU)",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_loaded": get_inference_engine() is not None
    }


@app.post("/dehaze")
async def dehaze_image(file: UploadFile = File(...)):
    """
    Dehaze a single image and return JSON with base64 data.
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, etc.)"
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Get inference engine
        engine = get_inference_engine()
        
        # Perform dehazing
        dehazed_image, haze_level = engine.dehaze_image(image_bytes)
        
        # Convert to base64
        output_buffer = io.BytesIO()
        dehazed_image.save(output_buffer, format='PNG')
        base64_img = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        
        return {
            "image_base64": base64_img,
            "haze_level": haze_level,
            "status": "success"
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.post("/dehaze/video")
async def dehaze_video(file: UploadFile = File(...)):
    """
    Video dehazing endpoint (placeholder).
    
    This is a planned future extension. Currently returns a placeholder response.
    
    Future implementation will:
        - Process video frame-by-frame
        - Maintain temporal consistency
        - Use batch processing for efficiency
        - Support common video formats (MP4, AVI, etc.)
    """
    return JSONResponse(
        status_code=501,
        content={
            "status": "not_implemented",
            "message": "Video dehazing is coming soon!",
            "details": "This feature is planned for future release.",
            "current_capabilities": [
                "Image dehazing (fully functional)"
            ],
            "planned_features": [
                "Frame-by-frame video processing",
                "Temporal consistency",
                "Batch processing optimization",
                "Multiple video format support"
            ]
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
