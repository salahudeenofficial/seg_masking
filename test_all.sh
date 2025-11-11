#!/bin/bash
# Batch test script for all test images

echo "=========================================="
echo "Testing SegFormer Masking on All Images"
echo "=========================================="

# Create output directory
mkdir -p test_outputs

# Test jeans (lower body)
echo ""
echo "Testing Jeans (Lower Body)..."
for img in test_images/test_jeans_*.jpg; do
    if [ -f "$img" ]; then
        name=$(basename "$img" .jpg)
        echo "  Processing: $name"
        python mask.py --mask_type lower_body --imagepath "$img" --output "test_outputs/${name}_masked.jpg" --debug 2>&1 | grep -E "(DEBUG|Warning|Error|Masked image)" || true
    fi
done

# Test upper body
echo ""
echo "Testing Upper Body..."
for img in test_images/test_upper_*.jpg; do
    if [ -f "$img" ]; then
        name=$(basename "$img" .jpg)
        echo "  Processing: $name"
        python mask.py --mask_type upper_body --imagepath "$img" --output "test_outputs/${name}_masked.jpg" --debug 2>&1 | grep -E "(DEBUG|Warning|Error|Masked image)" || true
    fi
done

# Test women's images (try both upper and lower)
echo ""
echo "Testing Women's Images..."
for img in test_images/test_women_*.jpg; do
    if [ -f "$img" ]; then
        name=$(basename "$img" .jpg)
        echo "  Processing: $name (upper_body)"
        python mask.py --mask_type upper_body --imagepath "$img" --output "test_outputs/${name}_upper_masked.jpg" --debug 2>&1 | grep -E "(DEBUG|Warning|Error|Masked image)" || true
    fi
done

echo ""
echo "=========================================="
echo "Testing Complete!"
echo "Check test_outputs/ directory for results"
echo "=========================================="

