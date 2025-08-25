# PANTHER Challenge Submission Files

This directory contains all the necessary files for PANTHER Challenge submission using SegResNet model.

## 📁 File Structure

### Core Inference Files
- `inference.py` - Main Docker container execution script for Grand Challenge
- `inference_submission.py` - Core inference logic with volume restoration
- `models.py` - SegResNet model definition (cleaned for submission)
- `dataloader.py` - Data loading and preprocessing

### Docker Configuration
- `Dockerfile` - Docker container configuration
- `requirements_docker.txt` - Python package requirements for Docker
- `build_docker.sh` - Automated Docker build script

### Configuration & Metrics
- `config.yaml` - Model and training configuration
- `metrics.py` - Evaluation metrics calculation
- `panther_metrics.py` - PANTHER challenge specific metrics

### Testing & Documentation
- `test_inference.py` - Local inference testing script
- `SUBMISSION_GUIDE.md` - Complete submission guide in English

## 🚀 Quick Start

### 1. Prepare Your Trained Model
Ensure you have a trained SegResNet model checkpoint:
```
experiments/segresnet_[timestamp]/
├── best_checkpoint.pth
└── config.yaml
```

### 2. Local Testing
```bash
# Auto-detect model and config from experiments folder
python test_inference.py \
    --test_image path/to/test_image.mha \
    --output_dir ./test_output

# Or specify manually
python test_inference.py \
    --model experiments/segresnet_[timestamp]/best_checkpoint.pth \
    --config experiments/segresnet_[timestamp]/config.yaml \
    --test_image path/to/test_image.mha \
    --output_dir ./test_output
```

### 3. Build Docker Container
```bash
# Make script executable
chmod +x build_docker.sh

# Build container
./build_docker.sh segresnet
```

### 4. Network Isolation Test (Recommended)
```bash
# Test container without internet access (simulates Grand Challenge)
./test_network_isolation.sh
```

### 5A. Save Container for Upload
```bash
docker save panther-challenge:latest | gzip > panther-challenge-segresnet.tar.gz
```

### 5B. Alternative: Create Separate Model Package
```bash
# Create model tarball for separate upload (recommended for large models)
./create_model_tarball.sh
# This creates: panther-model-segresnet.tar.gz
```

## 🔧 Key Features

### SegResNet Model
- Simplified and stable 3D U-Net style architecture
- Skip connections with proper size matching
- Dropout for regularization
- Binary segmentation (background vs foreground)

### Volume Restoration
- ✅ **Original Shape**: Prediction matches input exactly
- ✅ **Original Spacing**: Voxel spacing preserved
- ✅ **Original Metadata**: Origin, direction maintained
- ✅ **File Format**: Output saved as .nii.gz

### PANTHER Compliance
- Sliding window inference for large volumes
- Memory-efficient processing
- Proper label remapping (0,1→0, 2→1)
- Grand Challenge container standards

## ⚠️ Important Notes

1. **Model Architecture**: Only SegResNet is supported in this submission
2. **Input Format**: Supports .mha and .nii.gz files
3. **Output Format**: Always saves as .nii.gz with restored metadata
4. **Binary Classification**: Converts 3-class to 2-class problem
5. **Memory Optimization**: Uses sliding window for GPU efficiency

## 📊 Model Performance

The SegResNet model provides:
- Stable training convergence
- Efficient GPU memory usage
- Robust segmentation performance
- Fast inference speed

## 🔗 References

- [PANTHER Challenge](https://panther.grand-challenge.org/)
- [Grand Challenge Documentation](https://grand-challenge.org/documentation/)
- [Docker Installation](https://docs.docker.com/get-docker/)

## 📞 Support

For detailed submission instructions, see `SUBMISSION_GUIDE.md`.

---

**Ready for PANTHER Challenge Submission! 🎯**