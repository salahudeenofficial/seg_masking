#!/bin/bash
# Setup script for SegFormer-based masking

set -e

echo "=========================================="
echo "Setting up SegFormer B5 Masking"
echo "=========================================="

# Allow specifying environment name via CONDA_ENV environment variable
# Usage: CONDA_ENV=StableVITON ./setup.sh
CONDA_ENV="${CONDA_ENV:-segmasking}"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
if [[ $(echo "$PYTHON_VERSION >= 3.13" | bc -l 2>/dev/null || echo "0") == "1" ]]; then
    echo "WARNING: Python 3.13 detected. This may cause compatibility issues."
    echo "Recommended: Use Python 3.10 or 3.11"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if conda is available
if command -v conda &> /dev/null; then
    ENV_NAME="$CONDA_ENV"
    if conda env list | grep -q "$ENV_NAME"; then
        echo "Activating existing $ENV_NAME environment..."
        eval "$(conda shell.bash hook)"
        conda activate $ENV_NAME
    else
        echo "Creating $ENV_NAME environment with Python 3.10..."
        eval "$(conda shell.bash hook)"
        conda create -n $ENV_NAME python=3.10 -y
        conda activate $ENV_NAME
    fi
else
    echo "Conda not found. Using system Python..."
    echo "WARNING: Make sure you're using Python 3.10 or 3.11"
fi

echo "Upgrading pip..."
pip install --upgrade pip

echo ""
echo "=========================================="
echo "Installing PyTorch first..."
echo "=========================================="

# Check current PyTorch version if installed
PYTORCH_INSTALLED=false
PYTORCH_VERSION=""
if python -c "import torch" 2>/dev/null; then
    PYTORCH_INSTALLED=true
    PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
    echo "Found existing PyTorch: $PYTORCH_VERSION"
    
    # Check if PyTorch version is >= 2.1.0
    PYTORCH_MAJOR=$(echo $PYTORCH_VERSION | cut -d. -f1)
    PYTORCH_MINOR=$(echo $PYTORCH_VERSION | cut -d. -f2)
    if [[ "$PYTORCH_MAJOR" -gt 2 ]] || [[ "$PYTORCH_MAJOR" -eq 2 && "$PYTORCH_MINOR" -ge 1 ]]; then
        echo "PyTorch version is sufficient (>= 2.1.0)"
        SKIP_PYTORCH_INSTALL=true
    else
        echo "PyTorch version is too old (< 2.1.0). Upgrading..."
        SKIP_PYTORCH_INSTALL=false
    fi
else
    echo "PyTorch not found. Installing..."
    SKIP_PYTORCH_INSTALL=false
fi

# Install or upgrade PyTorch if needed
if [[ "$SKIP_PYTORCH_INSTALL" != "true" ]]; then
    # Detect CUDA version
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
        echo "Detected CUDA version: $CUDA_VERSION"
        
        if [[ "$CUDA_VERSION" == "11.8" ]] || [[ "$CUDA_VERSION" == "11.7" ]]; then
            echo "Installing PyTorch >= 2.1.0 for CUDA 11.8..."
            pip install --upgrade "torch>=2.1.0" torchvision --index-url https://download.pytorch.org/whl/cu118
        elif [[ "$CUDA_VERSION" == "12.1" ]] || [[ "$CUDA_VERSION" == "12.0" ]] || [[ "$CUDA_VERSION" == "12.9" ]]; then
            echo "Installing PyTorch >= 2.1.0 for CUDA 12.1+..."
            pip install --upgrade "torch>=2.1.0" torchvision --index-url https://download.pytorch.org/whl/cu121
        else
            echo "Installing PyTorch >= 2.1.0 for CUDA 11.8 (default)..."
            pip install --upgrade "torch>=2.1.0" torchvision --index-url https://download.pytorch.org/whl/cu118
        fi
    else
        echo "No GPU detected. Installing CPU-only PyTorch >= 2.1.0..."
        pip install --upgrade "torch>=2.1.0" torchvision
    fi
fi

echo ""
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" || {
    echo "ERROR: PyTorch installation failed!"
    exit 1
}

# Verify PyTorch version is >= 2.1.0
PYTORCH_VER_CHECK=$(python -c "import torch; from packaging import version; print('OK' if version.parse(torch.__version__.split('+')[0]) >= version.parse('2.1.0') else 'FAIL')" 2>/dev/null || echo "FAIL")
if [[ "$PYTORCH_VER_CHECK" != "OK" ]]; then
    echo "WARNING: PyTorch version may be too old. Transformers requires PyTorch >= 2.1.0"
    echo "Current version: $(python -c 'import torch; print(torch.__version__)')"
fi

echo ""
echo "=========================================="
echo "Installing other requirements..."
echo "=========================================="
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To use SegFormer masking:"
if command -v conda &> /dev/null; then
    echo "  conda activate $ENV_NAME"
else
    echo "  source venv/bin/activate  # if using venv"
fi
echo "  python mask.py --mask_type upper_body --imagepath path/to/image.jpg"
echo "  python mask_all_images.py  # Batch process all images"
echo ""
echo "Note: SegFormer model will be downloaded automatically on first use"
echo "      from HuggingFace (requires internet connection)"
echo ""
echo "To use with existing environment:"
echo "  CONDA_ENV=YourEnvName ./setup.sh"

