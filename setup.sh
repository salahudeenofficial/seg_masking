#!/bin/bash
# Setup script for SegFormer-based masking

set -e

echo "=========================================="
echo "Setting up SegFormer B5 Masking"
echo "=========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Check if StableVITON environment exists
if conda env list | grep -q "StableVITON"; then
    echo "Activating StableVITON environment..."
    eval "$(conda shell.bash hook)"
    conda activate StableVITON
else
    echo "Creating StableVITON environment..."
    eval "$(conda shell.bash hook)"
    conda create -n StableVITON python=3.10 -y
    conda activate StableVITON
fi

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To use SegFormer masking:"
echo "  conda activate StableVITON"
echo "  python mask.py --mask_type upper_body --imagepath path/to/image.jpg"
echo ""
echo "Note: SegFormer model will be downloaded automatically on first use"
echo "      from HuggingFace (requires internet connection)"

