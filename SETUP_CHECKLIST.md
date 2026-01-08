# Setup Checklist

Use this checklist to ensure your weather-adaptive image dehazing system is properly configured.

## ✅ Pre-Installation Checklist

### Hardware Requirements
- [ ] NVIDIA GPU (RTX 3050 or better)
- [ ] 4GB+ VRAM available
- [ ] 8GB+ system RAM
- [ ] 5GB+ free disk space

### Software Requirements
- [ ] Windows OS (or Linux/macOS with CUDA support)
- [ ] Python 3.8 or higher installed
- [ ] Node.js 18 or higher installed
- [ ] NPM package manager installed
- [ ] NVIDIA GPU drivers installed
- [ ] CUDA Toolkit 11.0+ installed

### Verification Commands
```bash
# Check Python version
python --version  # Should be 3.8+

# Check Node.js version
node --version  # Should be 18+

# Check NPM version
npm --version

# Check CUDA
nvidia-smi  # Should show GPU info
```

## ✅ Installation Checklist

### 1. Training Environment
- [ ] Navigated to `training/` directory
- [ ] Installed requirements: `pip install -r requirements.txt`
- [ ] Verified PyTorch installation
- [ ] Verified CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Result should be `True`

### 2. Backend Environment
- [ ] Navigated to `backend/` directory
- [ ] Installed requirements: `pip install -r requirements.txt`
- [ ] Verified FastAPI installation
- [ ] Verified Uvicorn installation

### 3. Frontend Environment
- [ ] Navigated to `frontend/` directory
- [ ] Ran `npm install`
- [ ] Installation completed without errors
- [ ] `node_modules/` directory created

## ✅ Dataset Preparation Checklist

### Kaggle Setup
- [ ] Created Kaggle account
- [ ] Generated API token
- [ ] Placed `kaggle.json` in correct location:
  - Windows: `C:\Users\<YourUsername>\.kaggle\`
  - Linux/Mac: `~/.kaggle/`
- [ ] Installed Kaggle CLI: `pip install kaggle`

### Dataset Download
- [ ] Chosen a dataset from Kaggle or created synthetic data
- [ ] Downloaded dataset
- [ ] Extracted dataset files

### Dataset Organization
- [ ] Created `data/` directory in project root
- [ ] Created `data/hazy/` subdirectory
- [ ] Created `data/clear/` subdirectory
- [ ] Copied hazy images to `data/hazy/`
- [ ] Copied clear images to `data/clear/`
- [ ] Verified matching filenames in both directories
- [ ] Verified at least 100 image pairs

### Dataset Verification
```bash
# Run this to verify dataset
python -c "
import os
hazy = len([f for f in os.listdir('data/hazy') if f.endswith(('.jpg', '.png'))])
clear = len([f for f in os.listdir('data/clear') if f.endswith(('.jpg', '.png'))])
print(f'Hazy: {hazy}, Clear: {clear}')
print('✓ Ready!' if hazy == clear and hazy > 0 else '✗ Issue detected')
"
```

## ✅ Training Checklist

### Pre-Training
- [ ] Dataset prepared and verified
- [ ] GPU memory cleared (close other GPU apps)
- [ ] Created `models/` directory (will be auto-created)

### Training Execution
- [ ] Navigated to `training/` directory
- [ ] Started training: `python train.py --data_dir ../data --epochs 10 --batch_size 4`
- [ ] Training started without errors
- [ ] GPU utilization visible (check with `nvidia-smi`)
- [ ] Training progress showing

### Post-Training
- [ ] Training completed successfully
- [ ] Model saved: `models/dehaze_model_best.pth` exists
- [ ] Training curves saved: `models/training_curves.png` exists
- [ ] Reviewed training curves for convergence

## ✅ Backend Checklist

### Pre-Launch
- [ ] Training completed and model exists
- [ ] Backend dependencies installed
- [ ] Port 8000 is available

### Backend Launch
- [ ] Navigated to `backend/` directory
- [ ] Started server: `uvicorn main:app --reload --host 0.0.0.0 --port 8000`
- [ ] Server started without errors
- [ ] Model loaded successfully (check logs)
- [ ] CUDA initialized (check logs)

### Backend Verification
- [ ] Opened `http://localhost:8000` in browser
- [ ] API information displayed
- [ ] Opened `http://localhost:8000/health`
- [ ] Health check shows `"status": "healthy"`
- [ ] Health check shows `"cuda_available": true`
- [ ] Health check shows `"model_loaded": true`

## ✅ Frontend Checklist

### Pre-Launch
- [ ] Backend is running on port 8000
- [ ] Frontend dependencies installed
- [ ] Port 3000 is available

### Frontend Launch
- [ ] Navigated to `frontend/` directory
- [ ] Started dev server: `npm run dev`
- [ ] Server started without errors
- [ ] No compilation errors

### Frontend Verification
- [ ] Opened `http://localhost:3000` in browser
- [ ] Page loads successfully
- [ ] Header and title visible
- [ ] Upload component displayed
- [ ] No console errors in browser DevTools

## ✅ End-to-End Testing Checklist

### Image Upload Test
- [ ] Prepared a test hazy image
- [ ] Clicked or dragged image to upload area
- [ ] Image upload initiated
- [ ] Processing animation displayed
- [ ] No errors in browser console
- [ ] No errors in backend logs

### Processing Test
- [ ] Processing completed (within 1-5 seconds)
- [ ] Dehazed image displayed
- [ ] Original image displayed
- [ ] Comparison slider visible
- [ ] Haze level indicator shown

### Interaction Test
- [ ] Dragged comparison slider
- [ ] Slider moves smoothly
- [ ] Before/after comparison works
- [ ] Side-by-side view displayed

### Download Test
- [ ] Clicked "Download Dehazed Image" button
- [ ] File download initiated
- [ ] Downloaded file is valid JPEG
- [ ] Downloaded image opens correctly

### Reset Test
- [ ] Clicked "Upload New Image" button
- [ ] Interface reset to upload state
- [ ] Can upload another image
- [ ] Process repeats successfully

## ✅ System Verification

### Run Automated Test
- [ ] Navigated to project root
- [ ] Ran: `python test_system.py`
- [ ] All tests passed
- [ ] CUDA test: ✓ PASSED
- [ ] Dependencies test: ✓ PASSED
- [ ] Model architecture test: ✓ PASSED
- [ ] Dataset test: ✓ PASSED (or ⚠ if not needed yet)

## ✅ Performance Verification

### Training Performance (RTX 3050)
- [ ] Batch size 4 works without OOM
- [ ] ~8-12 images/second processing speed
- [ ] ~5-10 minutes per epoch (1000 images)
- [ ] GPU utilization >80% during training

### Inference Performance
- [ ] Single image processes in <2 seconds
- [ ] GPU memory usage <4GB
- [ ] No memory leaks (can process multiple images)
- [ ] Consistent performance across images

### Web Performance
- [ ] Page loads in <2 seconds
- [ ] Animations run at 60 FPS
- [ ] No lag during interactions
- [ ] Responsive on different screen sizes

## ✅ Documentation Review

### Read Documentation
- [ ] Read `README.md` - Project overview
- [ ] Read `QUICKSTART.md` - Setup guide
- [ ] Read `DATASET_SETUP.md` - Dataset preparation
- [ ] Read `ARCHITECTURE.md` - System architecture
- [ ] Read `PROJECT_SUMMARY.md` - Implementation summary

### Understand System
- [ ] Understand training pipeline
- [ ] Understand inference backend
- [ ] Understand web application
- [ ] Understand data flow
- [ ] Understand model architecture

## ✅ Troubleshooting Checklist

If you encounter issues, verify:

### CUDA Issues
- [ ] NVIDIA drivers installed and up to date
- [ ] CUDA Toolkit installed
- [ ] PyTorch installed with CUDA support
- [ ] GPU visible: `nvidia-smi` works
- [ ] CUDA available in Python: `torch.cuda.is_available()` returns `True`

### Training Issues
- [ ] Dataset properly organized
- [ ] Sufficient GPU memory (try batch_size=2)
- [ ] No other GPU applications running
- [ ] Correct Python version (3.8+)

### Backend Issues
- [ ] Model file exists: `models/dehaze_model_best.pth`
- [ ] Port 8000 not in use
- [ ] All dependencies installed
- [ ] CUDA available

### Frontend Issues
- [ ] Backend running on port 8000
- [ ] Port 3000 not in use
- [ ] All dependencies installed (`npm install`)
- [ ] No TypeScript errors
- [ ] CORS enabled in backend

### Connection Issues
- [ ] Backend URL correct in frontend (`http://localhost:8000`)
- [ ] Firewall not blocking connections
- [ ] Both servers running simultaneously

## 🎉 Success Criteria

Your system is fully operational when:

✅ Training completes successfully and saves model  
✅ Backend starts and loads model on GPU  
✅ Frontend loads without errors  
✅ Can upload and process images end-to-end  
✅ Results display with comparison slider  
✅ Can download dehazed images  
✅ All animations work smoothly  
✅ No errors in console or logs  

## 📝 Notes

- Keep backend and frontend running simultaneously for testing
- Use `Ctrl+C` to stop servers when done
- Check logs for detailed error messages
- Refer to documentation for detailed troubleshooting

## 🚀 Next Steps After Setup

Once everything is verified:

1. **Experiment**: Try different hazy images
2. **Optimize**: Tune hyperparameters for better results
3. **Extend**: Add new features (batch processing, etc.)
4. **Deploy**: Consider cloud deployment
5. **Share**: Show off your working system!

---

**Last Updated**: 2026-01-07  
**Version**: 1.0.0  
**Status**: Production Ready ✅
