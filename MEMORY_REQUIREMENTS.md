# Memory Requirements - Complete Guide

## Overview

This document covers both **GPU VRAM** and **System RAM** requirements for running SegFormer-based masking.

---

## GPU VRAM (Video Memory) Requirements

### Model Memory Footprint

| Component | VRAM Usage | Notes |
|-----------|------------|-------|
| **SegFormer B5 Model** | ~2-3 GB | Model weights + activations |
| **SegFormer Inference** | +1-2 GB | During forward pass |
| **OpenPose Model** | ~1-2 GB | Body pose detection |
| **Image Buffers** | ~0.5-1 GB | Per image in batch |
| **CUDA Overhead** | ~0.5 GB | Framework overhead |

### Total VRAM by Use Case

#### Single Image Processing
- **Minimum**: 6 GB VRAM
- **Recommended**: 8 GB VRAM
- **Comfortable**: 12 GB VRAM

#### Batch Processing (4 images)
- **Minimum**: 10 GB VRAM
- **Recommended**: 12 GB VRAM
- **Comfortable**: 16 GB VRAM

#### Batch Processing (8 images)
- **Minimum**: 16 GB VRAM
- **Recommended**: 20 GB VRAM
- **Comfortable**: 24 GB VRAM

### VRAM Usage Breakdown

```
Single Image:
├── SegFormer B5:        2-3 GB
├── OpenPose:            1-2 GB
├── Image Processing:    0.5 GB
├── CUDA Overhead:       0.5 GB
└── Total:               ~6 GB

Batch of 4:
├── SegFormer B5:        2-3 GB (shared)
├── OpenPose:            1-2 GB (shared)
├── Image Processing:    2 GB (4 × 0.5 GB)
├── Batch Buffers:       2-3 GB
├── CUDA Overhead:       0.5 GB
└── Total:              ~10-12 GB

Batch of 8:
├── SegFormer B5:        2-3 GB (shared)
├── OpenPose:            1-2 GB (shared)
├── Image Processing:    4 GB (8 × 0.5 GB)
├── Batch Buffers:       4-6 GB
├── CUDA Overhead:       0.5 GB
└── Total:              ~16-20 GB
```

---

## System RAM Requirements

### Base Requirements

| Component | RAM Usage | Notes |
|-----------|-----------|-------|
| **Python Runtime** | ~200-500 MB | Base Python + libraries |
| **PyTorch** | ~500 MB - 1 GB | Framework overhead |
| **Transformers Library** | ~200-500 MB | HuggingFace models cache |
| **Image Loading** | ~50-200 MB | Per image (depends on resolution) |
| **OS + System** | ~1-2 GB | Operating system overhead |

### Total System RAM by Use Case

#### Single Image Processing
- **Minimum**: 4 GB RAM
- **Recommended**: 8 GB RAM
- **Comfortable**: 16 GB RAM

#### Batch Processing
- **Minimum**: 8 GB RAM
- **Recommended**: 16 GB RAM
- **Comfortable**: 32 GB RAM

### RAM Usage Breakdown

```
Single Image:
├── OS + System:         1-2 GB
├── Python + PyTorch:   1-1.5 GB
├── Model Cache:        0.5-1 GB
├── Image Loading:      0.1-0.2 GB
└── Total:              ~4-6 GB

Batch Processing (100 images):
├── OS + System:         1-2 GB
├── Python + PyTorch:   1-1.5 GB
├── Model Cache:        0.5-1 GB
├── Image Queue:        2-4 GB
├── Results Buffer:     1-2 GB
└── Total:              ~8-12 GB
```

---

## Memory Optimization Tips

### VRAM Optimization

1. **Process Images Sequentially**
   ```python
   # Instead of batch processing, process one at a time
   for image in images:
       result = masked_image('upper_body', image)
   ```

2. **Clear CUDA Cache**
   ```python
   import torch
   torch.cuda.empty_cache()  # After processing batch
   ```

3. **Use Mixed Precision** (if supported)
   ```python
   # Reduces memory by ~50%
   with torch.cuda.amp.autocast():
       output = model(input)
   ```

4. **Reduce Image Resolution**
   ```python
   # Process at lower resolution, then upscale
   masked_image(..., width=512, height=768)  # Instead of 1024x1024
   ```

5. **Unload Models When Not Needed**
   ```python
   # Move model to CPU when idle
   model.cpu()
   torch.cuda.empty_cache()
   ```

### RAM Optimization

1. **Process in Chunks**
   ```python
   # Process 100 images at a time instead of all at once
   for chunk in chunks(images, 100):
       process_batch(chunk)
   ```

2. **Clear Image Cache**
   ```python
   import gc
   del images
   gc.collect()
   ```

3. **Use Generators**
   ```python
   # Instead of loading all images into memory
   def image_generator(path):
       for img_path in paths:
           yield Image.open(img_path)
   ```

4. **Save Results Immediately**
   ```python
   # Don't keep all results in memory
   result = masked_image(...)
   # Save immediately, don't accumulate
   ```

---

## Vast.ai Instance Recommendations

### Budget Option
- **GPU**: RTX 3060 12GB
- **RAM**: 16 GB
- **Cost**: ~$0.20-0.35/hour
- **Best for**: Testing, small batches

### Recommended Option
- **GPU**: RTX 3080 10GB or RTX 3090 24GB
- **RAM**: 32 GB
- **Cost**: ~$0.40-0.80/hour
- **Best for**: Production, medium batches

### High Performance Option
- **GPU**: RTX 4090 24GB or A100 40GB
- **RAM**: 64 GB
- **Cost**: ~$1.00-3.00/hour
- **Best for**: Large-scale processing

---

## Memory Monitoring

### Check VRAM Usage
```bash
# Real-time GPU memory monitoring
watch -n 1 nvidia-smi

# Or use Python
python -c "import torch; print(f'VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB')"
```

### Check RAM Usage
```bash
# Linux
free -h

# Python
import psutil
print(f"RAM: {psutil.virtual_memory().used / 1e9:.2f} GB / {psutil.virtual_memory().total / 1e9:.2f} GB")
```

---

## Troubleshooting Memory Issues

### Out of Memory (OOM) Errors

#### VRAM OOM
**Symptoms**: `CUDA out of memory` error

**Solutions**:
1. Reduce batch size
2. Process images one at a time
3. Reduce image resolution
4. Use GPU with more VRAM
5. Clear cache: `torch.cuda.empty_cache()`

#### RAM OOM
**Symptoms**: System becomes slow, swapping to disk

**Solutions**:
1. Process in smaller chunks
2. Don't load all images at once
3. Use generators instead of lists
4. Increase instance RAM
5. Close other applications

### Memory Leaks

**Check for leaks**:
```python
import torch
import gc

# Before processing
mem_before = torch.cuda.memory_allocated()

# Process image
result = masked_image(...)

# After processing
torch.cuda.empty_cache()
gc.collect()
mem_after = torch.cuda.memory_allocated()

if mem_after > mem_before * 1.1:  # 10% increase
    print("Possible memory leak!")
```

---

## Example Memory Usage Scenarios

### Scenario 1: Single Image (1080x1440)
- **VRAM**: ~6 GB
- **RAM**: ~4 GB
- **Processing Time**: ~2-3 seconds

### Scenario 2: Batch of 10 Images
- **VRAM**: ~12 GB
- **RAM**: ~8 GB
- **Processing Time**: ~15-20 seconds

### Scenario 3: Large Dataset (1000 images)
- **VRAM**: ~8 GB (sequential processing)
- **RAM**: ~12 GB (with chunking)
- **Processing Time**: ~30-45 minutes

---

## Summary

### Minimum Requirements
- **VRAM**: 8 GB
- **RAM**: 8 GB
- **GPU**: RTX 3060 or better

### Recommended Requirements
- **VRAM**: 12-16 GB
- **RAM**: 16-32 GB
- **GPU**: RTX 3080/3090 or better

### Optimal Requirements
- **VRAM**: 24 GB+
- **RAM**: 32-64 GB
- **GPU**: RTX 4090 or A100

