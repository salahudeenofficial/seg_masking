# Vast.ai GPU Recommendations for SegFormer Masking

## Recommended GPU Specifications

### Minimum Requirements
- **GPU**: NVIDIA GPU with CUDA support
- **VRAM**: 8GB minimum (12GB recommended)
- **CUDA Compute Capability**: 7.0+ (Volta architecture or newer)

### Recommended Configurations

#### Option 1: Budget-Friendly (Good Performance)
- **GPU**: RTX 3060, RTX 3060 Ti, or RTX 3070
- **VRAM**: 12GB
- **Price**: ~$0.20-0.35/hour
- **Performance**: Good for batch processing 1-2 images at a time
- **Why**: Excellent price/performance ratio, sufficient VRAM for SegFormer B5

#### Option 2: Balanced (Recommended)
- **GPU**: RTX 3080, RTX 3090, or RTX 4070
- **VRAM**: 10-24GB
- **Price**: ~$0.40-0.80/hour
- **Performance**: Can process 4-8 images in parallel
- **Why**: Best balance of speed and cost for production use

#### Option 3: High Performance
- **GPU**: RTX 4090, A100, or A6000
- **VRAM**: 24GB+
- **Price**: ~$1.00-3.00/hour
- **Performance**: Can process 10+ images in parallel
- **Why**: Maximum throughput for large-scale processing

### Specific Recommendations

#### For Testing/Small Batches
- **RTX 3060 12GB** or **RTX 3070 8GB**
- Cost-effective for initial testing
- Sufficient for processing individual images

#### For Production/Medium Batches
- **RTX 3080 10GB** or **RTX 3090 24GB**
- Good balance of speed and memory
- Can handle batch processing efficiently

#### For Large-Scale Processing
- **RTX 4090 24GB** or **A100 40GB**
- Maximum performance
- Best for processing entire datasets quickly

## Memory Requirements

### SegFormer B5 Model
- **Model Size**: ~350MB (downloaded automatically)
- **VRAM Usage**: ~2-4GB for inference
- **Batch Processing**: +1-2GB per additional image in batch

### Total VRAM Needed
- **Single Image**: ~4-6GB
- **Batch of 4**: ~6-8GB
- **Batch of 8**: ~12-16GB

## Vast.ai Search Filters

When searching on Vast.ai, use these filters:

```
GPU: NVIDIA RTX 3060 or better
VRAM: >= 12GB (recommended)
CUDA: >= 11.0
Price: < $0.50/hour (for budget option)
```

## Setup on Vast.ai Instance

1. **Connect to instance** via SSH
2. **Install dependencies**:
   ```bash
   git clone https://github.com/salahudeenofficial/seg_masking.git
   cd seg_masking
   pip install -r requirements.txt
   ```

3. **Verify GPU**:
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

4. **Test the model**:
   ```bash
   python mask.py --mask_type upper_body --imagepath test_image.jpg
   ```

## Cost Estimation

### Processing 1000 images:
- **RTX 3060**: ~$5-10 (slower but cheaper)
- **RTX 3080**: ~$8-15 (balanced)
- **RTX 4090**: ~$15-25 (fastest)

*Estimates based on ~2-5 seconds per image processing time*

## Tips for Vast.ai

1. **Spot Instances**: Use spot instances for 30-50% cost savings
2. **Auto-shutdown**: Set auto-shutdown after job completion
3. **Storage**: Use instance storage for temporary files, upload results to cloud
4. **Monitoring**: Monitor GPU utilization with `nvidia-smi -l 1`
5. **Batch Processing**: Process multiple images in parallel to maximize GPU utilization

## Troubleshooting

### Out of Memory Errors
- Reduce batch size
- Use smaller image resolution
- Choose GPU with more VRAM

### Slow Processing
- Ensure CUDA is properly installed
- Check GPU utilization with `nvidia-smi`
- Consider upgrading to faster GPU

### Model Download Issues
- Ensure internet connection on instance
- Models download automatically on first use (~350MB)

