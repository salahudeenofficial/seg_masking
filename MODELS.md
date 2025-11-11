# Available SegFormer B5 Pretrained Models

## Recommended Models

### 1. **matei-dorian/segformer-b5-finetuned-human-parsing** (Default - BEST!)
- **Dataset**: Fine-tuned for human parsing
- **Classes**: 19-20 human parsing classes
- **Status**: ✅ Ready to use - NO MAPPING NEEDED!
- **Pros**: Actually trained for human parsing - most accurate results
- **Cons**: None - this is the best option!
- **Usage**: Already configured as default
- **URL**: https://huggingface.co/matei-dorian/segformer-b5-finetuned-human-parsing

### 2. **nvidia/segformer-b5-finetuned-ade-640-640** (Fallback)
- **Dataset**: ADE20K
- **Classes**: 150 semantic classes
- **Status**: ⚠️ Requires heuristic mapping
- **Pros**: General semantic segmentation model
- **Cons**: Requires mapping from 150 classes to 19 ATR classes (less accurate)
- **Usage**: Used as fallback if human parsing model unavailable

### 2. **nvidia/segformer-b5-finetuned-cityscapes-1024-1024**
- **Dataset**: Cityscapes (urban scenes)
- **Classes**: 19 classes (but for urban scenes, not human parsing)
- **Status**: ⚠️ Not suitable for human parsing
- **Note**: Has 19 classes but they're for urban scene segmentation (road, building, etc.)

### 3. **nvidia/mit-b5**
- **Type**: Base encoder only
- **Status**: ❌ Requires fine-tuning
- **Note**: Pre-trained on ImageNet, needs fine-tuning for segmentation

## Model Selection

The current implementation uses **nvidia/segformer-b5-finetuned-ade-640-640** by default because:
1. It's the best available general-purpose segmentation model
2. It has good person detection capabilities
3. We use heuristic mapping to convert to ATR labels

## Improving Results

For production use, consider:

1. **Fine-tune SegFormer B5 on human parsing dataset**:
   - Use ATR dataset (19 classes)
   - Or LIP dataset (20 classes)
   - Or Pascal-Person-Part dataset

2. **Use a pre-trained human parsing model** (if available):
   - Search HuggingFace for "human parsing" or "person parsing"
   - Look for models fine-tuned on ATR/LIP datasets

3. **Improve the mapping function**:
   - Current `_map_ade20k_to_atr()` uses simple heuristics
   - Can be improved with better spatial reasoning or ML-based mapping

## Download

Models are automatically downloaded from HuggingFace on first use. No manual download needed.

## Model URLs

- ADE20K: https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640
- Cityscapes: https://huggingface.co/nvidia/segformer-b5-finetuned-cityscapes-1024-1024
- Base: https://huggingface.co/nvidia/mit-b5

