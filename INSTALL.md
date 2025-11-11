# Installation Guide

## Important: Python Version

**Use Python 3.10 or 3.11** - Python 3.13 is too new and many packages don't support it yet.

## Quick Installation

### Step 1: Create Virtual Environment (Python 3.10/3.11)

```bash
# Using conda (recommended)
conda create -n segmasking python=3.10 -y
conda activate segmasking

# OR using venv
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### Step 2: Install PyTorch First

**For CUDA 11.8 (most common on Vast.ai):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only:**
```bash
pip install torch torchvision
```

**Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 3: Install Other Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "from transformers import SegformerForSemanticSegmentation; print('SegFormer OK')"
python -c "import torch; import cv2; import PIL; print('All dependencies OK')"
```

## Troubleshooting

### Error: "Skip building ext ops due to the absence of torch"

**Solution**: Install PyTorch BEFORE other packages:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Error: Python 3.13 compatibility issues

**Solution**: Use Python 3.10 or 3.11:
```bash
conda create -n segmasking python=3.10 -y
conda activate segmasking
```

### Error: mmcv-full build fails

**Solution**: mmcv-full is not needed for SegFormer. It's been removed from requirements.txt.
If you really need it, install after torch:
```bash
pip install torch torchvision
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```

### Error: CUDA not found

**Solution**: Check your CUDA version:
```bash
nvidia-smi
# Then install matching PyTorch version
# CUDA 11.8 -> cu118
# CUDA 12.1 -> cu121
```

## Installation on Vast.ai

### Complete Setup Script

```bash
# 1. Create environment
conda create -n segmasking python=3.10 -y
conda activate segmasking

# 2. Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Clone and install
git clone https://github.com/salahudeenofficial/seg_masking.git
cd seg_masking
pip install -r requirements.txt

# 4. Test
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python mask.py --mask_type upper_body --imagepath test.jpg
```

## Dependencies Breakdown

### Required
- **torch**: Deep learning framework
- **torchvision**: Image processing utilities
- **transformers**: HuggingFace library for SegFormer
- **Pillow**: Image loading/saving
- **opencv-python**: Image processing
- **numpy**: Numerical operations
- **einops**: Tensor operations (for OpenPose)

### Optional
- **timm**: Additional model utilities (not strictly required)
- **mmcv-full**: Advanced computer vision tools (not needed for SegFormer)

## Version Compatibility

| Component | Minimum | Recommended | Tested |
|-----------|---------|-------------|--------|
| Python | 3.8 | 3.10 | 3.10, 3.11 |
| PyTorch | 1.9.0 | 2.0.0+ | 2.0.0, 2.1.0 |
| CUDA | 11.0 | 11.8+ | 11.8, 12.1 |
| transformers | 4.20.0 | 4.30.0+ | 4.30.0+ |

## Common Issues

### Issue: "No module named 'transformers'"
```bash
pip install transformers
```

### Issue: "CUDA out of memory"
- Use smaller batch size
- Process images sequentially
- Use GPU with more VRAM

### Issue: "Model download fails"
- Check internet connection
- Models download automatically on first use (~350MB)
- May need to set HF_HOME environment variable

## Alternative: Using Pre-installed PyTorch

If PyTorch is already installed on your system:

```bash
# Just install other dependencies
pip install transformers pillow opencv-python numpy einops tqdm
```

