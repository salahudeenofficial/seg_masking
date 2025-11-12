# Mask SegFormer - Alternative Human Parsing with SegFormer B5

This is an alternative masking approach that uses **SegFormer B5** for human parsing instead of the ATR parsing model. Masking is performed entirely using SegFormer parsing - no OpenPose or other pose detection is required.

## Key Differences

- **Human Parsing**: Uses SegFormer B5 (state-of-the-art transformer-based segmentation)
- **Better Accuracy**: SegFormer typically provides better segmentation quality than ATR
- **Same Interface**: Drop-in replacement - same API as the original masking code

## Setup

1. Install dependencies:
```bash
conda activate StableVITON  # or your environment
pip install -r requirements.txt
```

2. Download SegFormer B5 model weights (will be done automatically on first run)

## Important Notes

### Model Fine-tuning Required

The current implementation uses a general SegFormer B5 model (trained on ADE20K). For optimal results, you should:

1. **Use a human parsing fine-tuned model**: If available, use a SegFormer model specifically fine-tuned for human parsing with 19 classes (matching ATR labels)

2. **Fine-tune yourself**: Fine-tune SegFormer B5 on a human parsing dataset (e.g., ATR, LIP, Pascal-Person-Part) to get 19-class output

3. **Class mapping**: The current code includes a placeholder `_map_to_atr_labels()` function that needs to be implemented based on your specific model

### Current Status

- ✅ SegFormer B5 integration structure complete
- ✅ Same API as original masking code
- ✅ No OpenPose dependencies required - pure SegFormer-based masking
- ⚠️ Class mapping needs implementation (or use fine-tuned model)

## Usage

Same as the original masking code:
```python
from mask import masked_image

result = masked_image(
    mask_type='upper_body',
    imagepath='path/to/image.jpg',
    output_path='output.jpg'
)
```

## Architecture

- **SegFormer B5**: For human parsing (replaces ATR parsing)
- **Masking Logic**: Rectangle mask creation from body parts detected by SegFormer

