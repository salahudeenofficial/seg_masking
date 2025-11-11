# Troubleshooting Guide

## Results Look Wrong

If the masking results don't look correct, here are common issues and solutions:

### Issue 1: Heuristic Mapping Limitations

**Problem**: The ADE20K model uses heuristic mapping which may not be accurate.

**Solution**: 
1. Use debug mode to see what's being detected:
   ```bash
   python mask.py --mask_type lower_body --imagepath test_jeans.jpg --output test.jpg --debug
   ```
   This will show you what labels are detected and save a parsing visualization.

2. **Best Solution**: Fine-tune SegFormer on a human parsing dataset (ATR, LIP, or Pascal-Person-Part)

### Issue 2: Person Not Detected

**Problem**: "Warning: No person detected in image"

**Solutions**:
- Check if image actually contains a person
- Try different image resolution
- The ADE20K model may not detect all person poses

### Issue 3: Wrong Body Parts Masked

**Problem**: Upper body masked when it should be lower body (or vice versa)

**Solutions**:
1. Check the parsing visualization (use `--debug` flag)
2. The heuristic thresholds may need adjustment for your images
3. Consider using a model fine-tuned for human parsing

### Issue 4: Mask Too Large/Small

**Problem**: Mask covers too much or too little area

**Solutions**:
- The bounding box approach creates rectangles - this is by design
- For more precise masks, you'd need pixel-perfect segmentation (requires fine-tuned model)

## Debug Mode

Enable debug mode to diagnose issues:

```bash
python mask.py --mask_type lower_body --imagepath test_jeans.jpg --output test.jpg --debug
```

This will:
1. Print detected labels and their pixel counts
2. Save a parsing visualization image showing what was detected
3. Help you understand what the model is seeing

## Improving Results

### Option 1: Fine-tune SegFormer (Best)

Fine-tune SegFormer B5 on a human parsing dataset:
- ATR dataset (19 classes)
- LIP dataset (20 classes)  
- Pascal-Person-Part dataset

This will give you accurate 19-class output without heuristic mapping.

### Option 2: Adjust Heuristic Thresholds

Edit `preprocess/humanparsing/run_parsing.py` and adjust the thresholds in `_map_ade20k_to_atr()`:
- `head_end`: Currently 12% of person height
- `upper_body_end`: Currently 55% of person height
- `lower_body_end`: Currently 90% of person height

### Option 3: Use Different Model

Try a different SegFormer model or look for human parsing specific models on HuggingFace.

## Expected Behavior

- **SegFormer detects person**: Should detect person regions (class 12 in ADE20K)
- **Heuristic mapping**: Divides person into face, upper body, lower body
- **Masking**: Creates rectangles around detected clothing regions
- **Output**: Green masked regions where clothing is detected

## Common Issues

### "No body parts detected"
- Person not detected by SegFormer
- Try different image or check if person is clearly visible

### "Mask is empty"
- No clothing regions detected
- Check debug output to see what labels were found

### Wrong regions masked
- Heuristic mapping may not work well for all poses
- Consider fine-tuning or adjusting thresholds

