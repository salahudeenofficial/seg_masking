# Instructions for Copying to Another System

## What to Copy

Copy the entire `mask_segformer` folder to your target system. This includes:

```
mask_segformer/
├── mask.py                    # Main masking function
├── mask_all_images.py         # Batch processing script
├── requirements.txt           # Dependencies
├── setup.sh                  # Setup script
├── test_all.sh               # Test script
├── preprocess/                # Preprocessing modules
│   └── humanparsing/
├── test_images/              # Test images
├── README.md                 # Documentation
├── INSTALL.md                # Installation guide
├── MODELS.md                 # Model information
├── MEMORY_REQUIREMENTS.md    # Memory specs
├── VAST_AI_SETUP.md          # Vast.ai guide
├── TROUBLESHOOTING.md        # Troubleshooting
└── QUICK_START.md            # Quick start guide
```

## Copy Methods

### Option 1: Using Git (Recommended)

```bash
# On target system
git clone https://github.com/salahudeenofficial/seg_masking.git
cd seg_masking
```

### Option 2: Using SCP

```bash
# From source system
scp -r mask_segformer user@target-system:/path/to/destination/
```

### Option 3: Using rsync

```bash
# From source system
rsync -avz mask_segformer/ user@target-system:/path/to/destination/mask_segformer/
```

### Option 4: Manual Copy

1. Zip the folder: `zip -r mask_segformer.zip mask_segformer/`
2. Transfer to target system
3. Unzip: `unzip mask_segformer.zip`

## Setup on Target System

After copying, follow these steps:

```bash
# 1. Navigate to the folder
cd mask_segformer

# 2. Create environment
conda create -n segmasking python=3.10 -y
conda activate segmasking

# 3. Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify installation
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "from transformers import SegformerForSemanticSegmentation; print('SegFormer OK')"
```

## Dataset Structure on Target System

Make sure your dataset is structured like this:

```
your_dataset/
├── combined/
│   ├── men/
│   │   ├── combined.csv
│   │   └── first_images/
│   └── women/
│       ├── combined.csv
│       └── first_images/
└── mask_segformer/          # The copied folder
    ├── mask.py
    ├── mask_all_images.py
    └── ...
```

## Running on Target System

```bash
# Activate environment
conda activate segmasking

# Navigate to mask_segformer
cd mask_segformer

# Run batch masking
python mask_all_images.py

# Or test first
python mask.py --mask_type lower_body --imagepath test_images/test_jeans_1.jpg --output test.jpg --debug
```

## Important Notes

1. **Dataset Path**: The script looks for dataset in `../combined/` relative to `mask_segformer` folder
   - If your dataset is elsewhere, modify `base_dir` in `mask_all_images.py`

2. **Model Download**: First run will download SegFormer model (~350MB)
   - Requires internet connection
   - Model cached in `~/.cache/huggingface/`

3. **GPU Required**: For best performance, use GPU
   - CPU will work but be very slow
   - Check GPU: `nvidia-smi`

4. **CSV Format**: Ensure your CSV has:
   - `image_filename` column (or `filename`)
   - `mask_type` column (`upper_body`, `lower_body`, or `other`)

## Quick Verification

After setup, test with:

```bash
# Test single image
python mask.py --mask_type lower_body --imagepath test_images/test_jeans_1.jpg --output test_output.jpg --debug

# Check output
ls -lh test_output.jpg
```

If this works, you're ready to process the full dataset!

