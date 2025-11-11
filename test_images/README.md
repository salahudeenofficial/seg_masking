# Test Images

This directory contains test images for validating the masking functionality.

## Image Categories

### Jeans (Lower Body)
- `test_jeans_1.jpg` - Jeans product image
- `test_jeans_2.jpg` - Jeans product image  
- `test_jeans_3.jpg` - Jeans product image

### Upper Body
- `test_upper_1.jpg` - Upper body clothing
- `test_upper_2.jpg` - Upper body clothing
- `test_upper_3.jpg` - Upper body clothing

### Women's Clothing
- `test_women_*.jpg` - Women's clothing images

## Testing Commands

### Test Lower Body Masking (Jeans)
```bash
python mask.py --mask_type lower_body --imagepath test_images/test_jeans_1.jpg --output test_output_jeans_1.jpg --debug
python mask.py --mask_type lower_body --imagepath test_images/test_jeans_2.jpg --output test_output_jeans_2.jpg --debug
python mask.py --mask_type lower_body --imagepath test_images/test_jeans_3.jpg --output test_output_jeans_3.jpg --debug
```

### Test Upper Body Masking
```bash
python mask.py --mask_type upper_body --imagepath test_images/test_upper_1.jpg --output test_output_upper_1.jpg --debug
python mask.py --mask_type upper_body --imagepath test_images/test_upper_2.jpg --output test_output_upper_2.jpg --debug
python mask.py --mask_type upper_body --imagepath test_images/test_upper_3.jpg --output test_output_upper_3.jpg --debug
```

### Batch Test Script
```bash
# Test all jeans images
for img in test_images/test_jeans_*.jpg; do
    python mask.py --mask_type lower_body --imagepath "$img" --output "${img%.jpg}_masked.jpg" --debug
done

# Test all upper body images
for img in test_images/test_upper_*.jpg; do
    python mask.py --mask_type upper_body --imagepath "$img" --output "${img%.jpg}_masked.jpg" --debug
done
```

## Expected Results

- **Lower body masking**: Should mask pants/jeans regions (labels 5, 6)
- **Upper body masking**: Should mask shirts/tops regions (label 4)
- **Debug output**: Shows detected labels and pixel counts
- **Parsing visualization**: Saved as `*_parsing_debug.png` when using `--debug`

