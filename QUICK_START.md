# Quick Start Guide - SegFormer Masking

## Setup (One Time)

```bash
# 1. Clone or copy the mask_segformer folder to your system
cd /path/to/your/workspace

# 2. Create conda environment
conda create -n segmasking python=3.10 -y
conda activate segmasking

# 3. Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Install other dependencies
cd mask_segformer
pip install -r requirements.txt
```

## Directory Structure Required

Make sure your dataset structure looks like this:

```
sample_dataset/
├── combined/
│   ├── men/
│   │   ├── combined.csv          # Must have 'mask_type' and 'image_filename' columns
│   │   ├── first_images/         # Source images
│   │   └── masked/               # Output folder (created automatically)
│   └── women/
│       ├── combined.csv
│       ├── first_images/
│       └── masked/
└── mask_segformer/              # This folder
    ├── mask.py
    ├── mask_all_images.py
    └── ...
```

## CSV Requirements

Your `combined.csv` files must have:
- `image_filename` column (or `filename`, or will try to find by `product_id`)
- `mask_type` column with values: `upper_body`, `lower_body`, or `other`

## Usage

### Option 1: Mask Entire Dataset (Recommended)

```bash
# Mask all images (men + women)
python mask_all_images.py

# Mask only men
python mask_all_images.py --gender men

# Mask only women
python mask_all_images.py --gender women

# Re-process all (even if masked exists)
python mask_all_images.py --no-skip

# With debug output
python mask_all_images.py --debug
```

### Option 2: Test on Single Image

```bash
# Test single image
python mask.py --mask_type lower_body --imagepath test_images/test_jeans_1.jpg --output test_output.jpg --debug
```

### Option 3: Batch Test Script

```bash
# Test all test images
./test_all.sh
```

## Output

- Masked images saved to: `combined/{gender}/masked/`
- Same filename as input (e.g., `25756738.jpg`)
- Green regions indicate masked clothing areas

## Monitoring Progress

The script shows:
- Progress bar for each gender
- Success/failure counts
- Summary at the end

## Troubleshooting

### "No module named 'mask'"
- Make sure you're in the `mask_segformer` directory
- Or add it to PYTHONPATH: `export PYTHONPATH=/path/to/mask_segformer:$PYTHONPATH`

### "CSV not found"
- Check that `combined/{gender}/combined.csv` exists
- Verify the path structure matches expected layout

### "Image not found"
- Check that images are in `combined/{gender}/first_images/`
- Verify `image_filename` column in CSV matches actual filenames

### Model download issues
- First run will download SegFormer model (~350MB)
- Requires internet connection
- Model cached after first download

## Performance

- **Processing speed**: ~2-5 seconds per image (depends on GPU)
- **1000 images**: ~30-80 minutes (depending on GPU)
- **Memory**: ~6-12 GB VRAM recommended

## Tips

1. **Start with small test**: Use `--gender men` first to test
2. **Use --debug**: See what labels are detected
3. **Check outputs**: Verify masked images look correct
4. **Monitor GPU**: Use `nvidia-smi` to check GPU usage
5. **Resume**: Script skips existing images by default (use `--no-skip` to reprocess)

