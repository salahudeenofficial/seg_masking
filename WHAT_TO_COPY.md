# What to Copy - Complete Guide

## Option 1: Copy Everything (Recommended for First Time)

Copy these folders to your target system:

### 1. The `mask_segformer` Folder (Entire Folder)
```
mask_segformer/
├── mask.py
├── mask_all_images.py          # Main batch script
├── requirements.txt
├── setup.sh
├── test_all.sh
├── preprocess/
│   ├── humanparsing/
│   │   └── run_parsing.py
│   └── openpose/
│       └── run_openpose.py
├── test_images/                # Optional - for testing
└── [all .md documentation files]
```

### 2. Your Dataset Structure
```
combined/
├── men/
│   ├── combined.csv            # Required: must have mask_type column
│   └── first_images/           # Required: source images
│       └── *.jpg
└── women/
    ├── combined.csv            # Required: must have mask_type column
    └── first_images/           # Required: source images
        └── *.jpg
```

**Note**: The `masked/` folders will be created automatically by the script.

## Option 2: Minimal Copy (Just Code)

If you already have the dataset on the target system, you only need:

```
mask_segformer/
├── mask.py
├── mask_all_images.py
├── requirements.txt
├── preprocess/
│   ├── humanparsing/
│   │   └── run_parsing.py
│   └── openpose/
│       └── run_openpose.py
└── __init__.py files
```

## Directory Structure on Target System

After copying, your structure should look like:

```
your_workspace/
├── combined/                   # Your dataset
│   ├── men/
│   │   ├── combined.csv
│   │   └── first_images/
│   └── women/
│       ├── combined.csv
│       └── first_images/
└── mask_segformer/            # The copied folder
    ├── mask.py
    ├── mask_all_images.py
    └── ...
```

## Quick Copy Commands

### Using Git (Easiest)
```bash
# On target system
git clone https://github.com/salahudeenofficial/seg_masking.git
cd seg_masking
# Rename if needed
mv seg_masking mask_segformer
```

### Using SCP
```bash
# From source system
scp -r mask_segformer/ user@target:/path/to/workspace/
scp -r combined/ user@target:/path/to/workspace/
```

### Using rsync (Best for large datasets)
```bash
# From source system
rsync -avz --progress mask_segformer/ user@target:/path/to/workspace/mask_segformer/
rsync -avz --progress combined/ user@target:/path/to/workspace/combined/
```

### Using Zip/Tar
```bash
# On source system
tar -czf mask_segformer.tar.gz mask_segformer/
tar -czf combined.tar.gz combined/

# Transfer files, then on target system
tar -xzf mask_segformer.tar.gz
tar -xzf combined.tar.gz
```

## File Sizes (Approximate)

- `mask_segformer/`: ~2-5 MB (code only)
- `combined/`: Depends on your dataset size
  - CSVs: Very small (< 1 MB)
  - Images: Can be large (GBs)

## What NOT to Copy

You don't need to copy:
- `__pycache__/` folders (will be regenerated)
- `.git/` folder (unless you want version control)
- `test_outputs/` (if exists, will be regenerated)
- Model checkpoints (will download automatically)
- Virtual environments (create new one on target)

## Verification After Copy

On target system, verify:

```bash
# Check mask_segformer structure
ls mask_segformer/
# Should see: mask.py, mask_all_images.py, requirements.txt, preprocess/

# Check dataset structure
ls combined/men/
# Should see: combined.csv, first_images/

# Check CSV has required columns
head -1 combined/men/combined.csv
# Should include: mask_type, image_filename (or similar)
```

## Quick Test After Copy

```bash
cd mask_segformer
conda activate segmasking
python mask.py --mask_type lower_body --imagepath test_images/test_jeans_1.jpg --output test.jpg
```

If this works, you're ready to process the full dataset!

