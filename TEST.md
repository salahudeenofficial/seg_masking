# Testing Guide

## Quick Test

After installation, test the masking with the included test image:

```bash
# Test upper body masking
python mask.py --mask_type upper_body --imagepath test_jeans.jpg --output test_upper_output.jpg

# Test lower body masking (for jeans)
python mask.py --mask_type lower_body --imagepath test_jeans.jpg --output test_lower_output.jpg
```

## Expected Output

- **Input**: `test_jeans.jpg` - A jeans product image
- **Output**: Masked image with green regions where clothing is detected
- **Processing time**: ~2-5 seconds per image (depending on GPU)

## Verify Installation

```bash
# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check SegFormer
python -c "from transformers import SegformerForSemanticSegmentation; print('SegFormer OK')"

# Check all dependencies
python -c "import torch, transformers, PIL, cv2, numpy; print('All dependencies OK')"
```

## Troubleshooting

### If you get "CUDA out of memory"
- The test image is 1080x1440 - this should work on 8GB+ VRAM
- If issues persist, reduce resolution in the code

### If model download fails
- Check internet connection
- Models download automatically on first use (~350MB)
- May take a few minutes on slow connections

### If parsing fails
- Check that SegFormer model downloaded correctly
- Verify GPU is available: `nvidia-smi`
- Check logs for specific error messages

