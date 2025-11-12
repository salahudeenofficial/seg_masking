#!/usr/bin/env python3
"""
Masking script using SegFormer B5 for human parsing.
Alternative approach to mask_gpu that uses SegFormer instead of ATR parsing.
"""

import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

# Add preprocess directories to path
project_root = Path(__file__).absolute().parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "preprocess" / "humanparsing"))

# Global cache for preprocessors to avoid re-initialization
_preprocessors_cache = None


def get_clothing_bounding_box(model_parse: Image.Image, category: str, width: int, height: int):
    """
    Get the bounding box coordinates of clothing parts detected in the parsing result.
    Returns the combined bounding box that covers all relevant clothing parts.
    
    Returns:
        tuple: (x_min, y_min, x_max, y_max) or None if no clothing detected
    """
    parse_array = np.array(model_parse.resize((width, height), Image.NEAREST))
    
    # Exclude shoe/feet labels: 9 (left_shoe), 10 (right_shoe)
    excluded_labels = {9, 10}  # Never include shoes
    
    # Define body parts to mask based on category (excluding shoes/feet)
    # Note: Label 7 (dress) is a full-body garment and should NOT be included in 
    # upper_body or lower_body masking - it would mask the entire body
    if category == "upper_body":
        body_part_labels = [4]  # upper_clothes only (exclude dress 7 - it's full body)
    elif category == "lower_body":
        body_part_labels = [5, 6]  # skirt, pants (excluded legs 12,13 to avoid feet, exclude dress 7)
    elif category == "dresses":
        body_part_labels = [4, 5, 6, 7]  # upper_clothes, skirt, pants, dress
    else:
        body_part_labels = [4, 5, 6, 7]  # default: all clothing parts
    
    # Remove any excluded labels if they accidentally appear
    body_part_labels = [label for label in body_part_labels if label not in excluded_labels]
    
    # Collect all pixels from all relevant body parts
    all_rows = []
    all_cols = []
    
    for label in body_part_labels:
        part_mask = (parse_array == label).astype(np.uint8)
        if part_mask.sum() == 0:
            continue
        
        # Find pixels of this body part
        rows, cols = np.where(part_mask > 0)
        if len(rows) > 0 and len(cols) > 0:
            all_rows.extend(rows)
            all_cols.extend(cols)
    
    # If no clothing detected, return None
    if len(all_rows) == 0 or len(all_cols) == 0:
        return None
    
    # Get combined bounding box
    y_min, y_max = int(min(all_rows)), int(max(all_rows))
    x_min, x_max = int(min(all_cols)), int(max(all_cols))
    
    # Ensure bounding box stays within image bounds
    rect_x_min = max(0, x_min)
    rect_x_max = min(width, x_max + 1)
    rect_y_min = max(0, y_min)
    rect_y_max = min(height, y_max + 1)
    
    # Add some padding (5% on each side, minimum 10 pixels)
    padding_x = max(10, int((rect_x_max - rect_x_min) * 0.05))
    padding_y = max(10, int((rect_y_max - rect_y_min) * 0.05))
    
    rect_x_min = max(0, rect_x_min - padding_x)
    rect_x_max = min(width, rect_x_max + padding_x)
    rect_y_min = max(0, rect_y_min - padding_y)
    rect_y_max = min(height, rect_y_max + padding_y)
    
    if rect_x_max > rect_x_min and rect_y_max > rect_y_min:
        return (rect_x_min, rect_y_min, rect_x_max, rect_y_max)
    
    return None


def resize_panel(img: Image.Image, width: int, height: int) -> Image.Image:
    """Resize image to specified dimensions."""
    return img.convert("RGB").resize((width, height))


def init_preprocessors(use_gpu_device: int = 0):
    """
    Initialize SegFormer Parsing preprocessor.
    
    Only SegFormer parsing is required for masking.
    """
    try:
        from preprocess.humanparsing.run_parsing import SegFormerParsing
    except Exception as e:
        raise RuntimeError(f"Preprocessors not available: {e}")
    
    # SegFormer parsing is REQUIRED for masking
    parsing = SegFormerParsing(use_gpu_device)
    
    return parsing


def get_preprocessors(use_gpu_device: int = 0):
    """Get preprocessors, using cache if available."""
    global _preprocessors_cache
    if _preprocessors_cache is None:
        _preprocessors_cache = init_preprocessors(use_gpu_device)
    return _preprocessors_cache


def save_image(img: Image.Image, path: str):
    """Save image to path, creating directories if needed."""
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    ext = Path(path).suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        img.convert("RGB").save(path, format="JPEG", quality=95)
    elif ext == ".png":
        img.save(path, format="PNG")
    else:
        img.convert("RGB").save(path, format="PNG")  # Default to PNG


def masked_image(mask_type: str, imagepath: str, output_path: str = None, 
                 width: int = 576, height: int = 768, device_index: int = 0, 
                 preserve_resolution: bool = True, debug: bool = False) -> str:
    """
    Crop clothing region from image based on mask_type using SegFormer B5 for parsing.
    
    Args:
        mask_type: One of 'upper_body', 'lower_body', or 'other'
        imagepath: Path to the input image
        output_path: Optional path for output image. If None, saves next to input with '_cropped' suffix
        width: Working resolution width for processing (default: 576). Only used if preserve_resolution=False
        height: Working resolution height for processing (default: 768). Only used if preserve_resolution=False
        device_index: GPU device index (default: 0)
        preserve_resolution: If True, output will be cropped from original resolution. If False, uses width/height (default: True)
    
    Returns:
        Path to the cropped clothing image (or original image path if mask_type is 'other')
    
    Raises:
        FileNotFoundError: If input image doesn't exist
        ValueError: If mask_type is not one of the valid options
        RuntimeError: If preprocessors cannot be initialized
    """
    # Validate mask_type
    valid_mask_types = ['upper_body', 'lower_body', 'other']
    if mask_type not in valid_mask_types:
        raise ValueError(f"mask_type must be one of {valid_mask_types}, got '{mask_type}'")
    
    # Validate input image exists
    image_path = Path(imagepath)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {imagepath}")
    
    # For 'other' type, just return the original image path (bypass)
    if mask_type == "other":
        return str(image_path)
    
    # Load the image
    try:
        person_img = Image.open(image_path)
        original_width, original_height = person_img.size
    except Exception as e:
        raise RuntimeError(f"Error loading image {imagepath}: {e}")
    
    # Get preprocessors
    parsing = get_preprocessors(use_gpu_device=device_index)
    
    # Determine processing dimensions
    if preserve_resolution:
        # Use working resolution for processing, but we'll scale back to original
        process_width, process_height = width, height
    else:
        # Use specified dimensions for both processing and output
        process_width, process_height = width, height
        original_width, original_height = width, height
    
    # Resize image for processing (models work better at standard sizes)
    person_r = resize_panel(person_img, process_width, process_height)
    
    # Apply masking
    try:
        # SegFormer parsing is required for masking
        try:
            model_parse, face_mask = parsing(person_r)
        except Exception as e:
            print(f"ERROR: SegFormer parsing failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Check if parsing result is valid
        if model_parse is None:
            raise ValueError("Parsing returned None")
        
        # Debug: Save parsing visualization
        if debug:
            parse_array = np.array(model_parse)
            unique_labels = np.unique(parse_array)
            print(f"\n{'='*60}")
            print(f"DEBUG: Parsing Analysis")
            print(f"{'='*60}")
            print(f"Detected labels: {unique_labels}")
            print(f"Label counts:")
            # Get label mapping from parsing class
            try:
                label_mapping = parsing.LABEL_MAPPING
            except AttributeError:
                label_mapping = {}
            
            for label in unique_labels:
                count = np.sum(parse_array == label)
                label_name = label_mapping.get(int(label), f"unknown_{label}")
                print(f"  Label {label:2d} ({label_name:15s}): {count:8d} pixels")
            print(f"{'='*60}\n")
            
            # Save parsing visualization
            debug_path = Path(output_path).parent / f"{Path(output_path).stem}_parsing_debug.png" if output_path else Path(imagepath).parent / f"{Path(imagepath).stem}_parsing_debug.png"
            model_parse.save(str(debug_path))
            print(f"Debug: Parsing visualization saved to {debug_path}")
        
        # Get bounding box of clothing at processing resolution
        bbox = get_clothing_bounding_box(
            model_parse=model_parse,
            category=mask_type,
            width=process_width,
            height=process_height,
        )
        
        # Check if clothing detected
        if bbox is None:
            raise ValueError("No clothing parts detected in parsing result")
        
        x_min, y_min, x_max, y_max = bbox
        
        # Scale bounding box to original resolution if preserving resolution
        if preserve_resolution and (process_width != original_width or process_height != original_height):
            scale_x = original_width / process_width
            scale_y = original_height / process_height
            x_min = int(x_min * scale_x)
            y_min = int(y_min * scale_y)
            x_max = int(x_max * scale_x)
            y_max = int(y_max * scale_y)
            # Ensure within bounds
            x_min = max(0, min(x_min, original_width))
            y_min = max(0, min(y_min, original_height))
            x_max = max(x_min, min(x_max, original_width))
            y_max = max(y_min, min(y_max, original_height))
        else:
            # Use original dimensions if not preserving resolution
            original_width, original_height = process_width, process_height
        
        # Use original image at original resolution for cropping
        if preserve_resolution:
            person_final = person_img.convert('RGB')
        else:
            person_final = person_r.convert('RGB')
        
        # Crop the clothing region
        if x_max > x_min and y_max > y_min:
            cropped_clothing = person_final.crop((x_min, y_min, x_max, y_max))
        else:
            raise ValueError(f"Invalid bounding box: ({x_min}, {y_min}, {x_max}, {y_max})")
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_type = type(e).__name__
        print(f"\n{'='*60}")
        print(f"ERROR: Could not crop clothing region")
        print(f"Error type: {error_type}")
        print(f"Error message: {error_msg}")
        print(f"{'='*60}")
        print("Full traceback:")
        traceback.print_exc()
        print(f"{'='*60}\n")
        
        # Re-raise the exception to see the full error
        raise RuntimeError(f"Clothing cropping failed: {error_type}: {error_msg}") from e
    
    # Determine output path
    if output_path is None:
        # Save next to original with '_cropped' suffix
        output_path = image_path.parent / f"{image_path.stem}_cropped{image_path.suffix}"
    else:
        output_path = Path(output_path)
    
    # Save the cropped clothing image
    save_image(cropped_clothing, str(output_path))
    
    if debug:
        print(f"Cropped clothing region: ({x_min}, {y_min}) to ({x_max}, {y_max})")
        print(f"Cropped image size: {cropped_clothing.size[0]}x{cropped_clothing.size[1]}")
    
    return str(output_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Crop clothing region from image using SegFormer B5")
    parser.add_argument("--mask_type", choices=['upper_body', 'lower_body', 'other'],
                       help="Type of mask to apply")
    parser.add_argument("--imagepath", help="Path to input image")
    parser.add_argument("--output", help="Path to output image (default: input_path_masked.ext)")
    parser.add_argument("--width", type=int, default=576, help="Working resolution width for processing (default: 576)")
    parser.add_argument("--height", type=int, default=768, help="Working resolution height for processing (default: 768)")
    parser.add_argument("--device", type=int, default=0, help="GPU device index (default: 0)")
    parser.add_argument("--preserve_resolution", action="store_true", default=True,
                       help="Preserve input image resolution in output (default: True)")
    parser.add_argument("--no_preserve_resolution", dest="preserve_resolution", action="store_false",
                       help="Use fixed width/height for output instead of preserving input resolution")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode: show detected labels and save parsing visualization")
    
    args = parser.parse_args()
    
    try:
        result_path = masked_image(
            mask_type=args.mask_type,
            imagepath=args.imagepath,
            output_path=args.output,
            width=args.width,
            height=args.height,
            device_index=args.device,
            preserve_resolution=args.preserve_resolution,
            debug=args.debug
        )
        print(f"Cropped clothing image saved to: {result_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

