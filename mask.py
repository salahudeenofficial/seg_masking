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


def get_clothing_bounding_box(model_parse: Image.Image, category: str, width: int, height: int, debug: bool = False):
    """
    Get precise bounding box coordinates of clothing parts detected in the parsing result.
    Returns a tight bounding box that minimizes non-target areas.
    
    Returns:
        tuple: (x_min, y_min, x_max, y_max) or None if no clothing detected
    """
    # Resize parse to processing dimensions
    parse_resized = model_parse.resize((width, height), Image.NEAREST)
    parse_array = np.array(parse_resized)
    
    if debug:
        print(f"DEBUG: parse_array shape: {parse_array.shape}, unique labels: {np.unique(parse_array)}")
    
    # Exclude shoe/feet labels: 9 (left_shoe), 10 (right_shoe)
    excluded_labels = {9, 10}  # Never include shoes
    
    # Define body parts to mask based on category (excluding shoes/feet)
    # Note: Label 7 (dress) is a full-body garment and should NOT be included in 
    # upper_body or lower_body masking - it would mask the entire body
    if category == "upper_body":
        target_labels = [4]  # upper_clothes only (exclude dress 7 - it's full body)
        # STRICTLY exclude lower_body labels (5=skirt, 6=pants) and all other non-target labels
        exclude_labels = {0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}
        opposite_labels = [5, 6]  # Explicitly exclude lower body labels
    elif category == "lower_body":
        target_labels = [5, 6]  # skirt, pants (excluded legs 12,13 to avoid feet, exclude dress 7)
        # STRICTLY exclude upper_body label (4=upper_clothes) and all other non-target labels
        exclude_labels = {0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}
        opposite_labels = [4]  # Explicitly exclude upper body label
    elif category == "dresses":
        target_labels = [4, 5, 6, 7]  # upper_clothes, skirt, pants, dress
        exclude_labels = {9, 10}  # Only exclude shoes
        opposite_labels = []  # No opposite for dresses
    else:
        target_labels = [4, 5, 6, 7]  # default: all clothing parts
        exclude_labels = {9, 10}  # Only exclude shoes
        opposite_labels = []  # No opposite for default
    
    # Remove any excluded labels if they accidentally appear
    target_labels = [label for label in target_labels if label not in excluded_labels]
    
    # Create a mask for target clothing parts only
    # STRICTLY exclude opposite category labels from the mask
    target_mask = np.zeros((height, width), dtype=np.uint8)
    for label in target_labels:
        label_mask = (parse_array == label).astype(np.uint8)
        target_mask |= label_mask
    
    # STRICTLY remove opposite category labels from the mask
    # This ensures that pixels labeled as opposite category are excluded from bounding box
    if opposite_labels:
        opposite_mask = np.zeros((height, width), dtype=bool)
        for label in opposite_labels:
            opposite_mask |= (parse_array == label)
        # Remove opposite category pixels from target mask
        target_mask = np.where(opposite_mask, 0, target_mask).astype(np.uint8)
    
    # If no target clothing detected, return None
    target_pixel_count = target_mask.sum()
    if debug:
        print(f"DEBUG: target_labels: {target_labels}, target_pixel_count: {target_pixel_count}")
        if opposite_labels:
            opposite_pixel_count = np.sum(np.isin(parse_array, opposite_labels))
            print(f"DEBUG: opposite_labels excluded: {opposite_labels}, opposite_pixel_count: {opposite_pixel_count}")
            print(f"DEBUG: target_pixel_count after exclusion: {target_pixel_count}")
    if target_pixel_count == 0:
        if debug:
            print(f"DEBUG: No target pixels found after exclusions, returning None")
        return None
    
    # Find bounding box of target clothing only
    rows, cols = np.where(target_mask > 0)
    if debug:
        print(f"DEBUG: Found {len(rows)} target pixels for bounding box")
    if len(rows) == 0 or len(cols) == 0:
        if debug:
            print(f"DEBUG: No rows/cols found, returning None")
        return None
    
    # Get initial bounding box from target clothing pixels
    y_min, y_max = int(rows.min()), int(rows.max())
    x_min, x_max = int(cols.min()), int(cols.max())
    
    # Refine bounding box by filtering out areas with too many non-target labels
    # This ensures the cropped region is mostly target clothing
    bbox_height = y_max - y_min
    bbox_width = x_max - x_min
    
    # Calculate target label density in the bounding box region
    bbox_region = parse_array[y_min:y_max+1, x_min:x_max+1]
    target_pixels_in_bbox = np.sum(np.isin(bbox_region, target_labels))
    total_pixels_in_bbox = bbox_region.size
    target_density = target_pixels_in_bbox / total_pixels_in_bbox if total_pixels_in_bbox > 0 else 0
    
    # If target density is too low, try to shrink the bounding box
    # by removing rows/columns with low target density
    # But only if we have enough pixels to work with
    if target_density < 0.3 and target_pixel_count > 100:  # Only refine if we have enough pixels
        # Try to find a tighter bounding box
        # Find rows and columns with sufficient target density
        row_target_counts = np.sum(target_mask, axis=1)
        col_target_counts = np.sum(target_mask, axis=0)
        
        # Find rows with target pixels (at least 1 pixel, or 1% of row width, whichever is larger)
        min_cols_per_row = max(1, int(width * 0.01))
        valid_rows = np.where(row_target_counts >= min_cols_per_row)[0]
        
        # Find columns with target pixels (at least 1 pixel, or 1% of column height, whichever is larger)
        min_rows_per_col = max(1, int(height * 0.01))
        valid_cols = np.where(col_target_counts >= min_rows_per_col)[0]
        
        if len(valid_rows) > 0 and len(valid_cols) > 0:
            y_min = int(valid_rows.min())
            y_max = int(valid_rows.max())
            x_min = int(valid_cols.min())
            x_max = int(valid_cols.max())
            if debug:
                print(f"DEBUG: Refined bbox from ({x_min}, {y_min}, {x_max}, {y_max})")
    
    # Ensure bounding box stays within image bounds
    rect_x_min = max(0, x_min)
    rect_x_max = min(width, x_max + 1)
    rect_y_min = max(0, y_min)
    rect_y_max = min(height, y_max + 1)
    
    # Add minimal padding (only 2-3 pixels) to avoid cutting off edges
    # Much smaller than before to minimize non-target areas
    padding_x = 3
    padding_y = 3
    
    rect_x_min = max(0, rect_x_min - padding_x)
    rect_x_max = min(width, rect_x_max + padding_x)
    rect_y_min = max(0, rect_y_min - padding_y)
    rect_y_max = min(height, rect_y_max + padding_y)
    
    # Final validation: ensure the bounding box is valid
    # Check if we have any target pixels in the final bounding box
    final_bbox_region = parse_array[rect_y_min:rect_y_max, rect_x_min:rect_x_max]
    final_target_pixels = np.sum(np.isin(final_bbox_region, target_labels))
    
    if debug:
        print(f"DEBUG: Final bbox: ({rect_x_min}, {rect_y_min}, {rect_x_max}, {rect_y_max})")
        print(f"DEBUG: Final target pixels in bbox: {final_target_pixels}")
    
    # Return bounding box if valid dimensions and has target pixels
    if rect_x_max > rect_x_min and rect_y_max > rect_y_min and final_target_pixels > 0:
        return (rect_x_min, rect_y_min, rect_x_max, rect_y_max)
    
    if debug:
        print(f"DEBUG: Bounding box validation failed, returning None")
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
        # If parsing failed, save original image as output
        if model_parse is None:
            if debug:
                print(f"Warning: Parsing returned None. Saving original image.")
            # Determine output path
            if output_path is None:
                output_path = image_path.parent / f"{image_path.stem}_cropped{image_path.suffix}"
            else:
                output_path = Path(output_path)
            
            # Save original image as fallback
            save_image(person_img.convert('RGB'), str(output_path))
            return str(output_path)
        
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
            debug=debug,
        )
        
        # Check if clothing detected
        # If no clothing detected, save original image as output
        if bbox is None:
            if debug:
                print(f"Warning: No clothing parts detected for {mask_type}. Saving original image.")
            # Determine output path
            if output_path is None:
                output_path = image_path.parent / f"{image_path.stem}_cropped{image_path.suffix}"
            else:
                output_path = Path(output_path)
            
            # Save original image as fallback
            save_image(person_img.convert('RGB'), str(output_path))
            return str(output_path)
        
        x_min, y_min, x_max, y_max = bbox
        
        # Calculate target label density for debug info
        if debug:
            parse_array_debug = np.array(model_parse.resize((process_width, process_height), Image.NEAREST))
            bbox_region_debug = parse_array_debug[y_min:y_max, x_min:x_max]
            
            if mask_type == "upper_body":
                target_labels_debug = [4]
            elif mask_type == "lower_body":
                target_labels_debug = [5, 6]
            else:
                target_labels_debug = [4, 5, 6, 7]
            
            target_pixels = np.sum(np.isin(bbox_region_debug, target_labels_debug))
            total_pixels = bbox_region_debug.size
            target_density = (target_pixels / total_pixels * 100) if total_pixels > 0 else 0
            
            print(f"\nBounding box at processing resolution: ({x_min}, {y_min}) to ({x_max}, {y_max})")
            print(f"Bounding box size: {x_max - x_min} x {y_max - y_min}")
            print(f"Target clothing density in bbox: {target_density:.1f}%")
        
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
        # If invalid bounding box, save original image as output
        if x_max > x_min and y_max > y_min:
            cropped_clothing = person_final.crop((x_min, y_min, x_max, y_max))
        else:
            if debug:
                print(f"Warning: Invalid bounding box: ({x_min}, {y_min}, {x_max}, {y_max}). Saving original image.")
            # Determine output path
            if output_path is None:
                output_path = image_path.parent / f"{image_path.stem}_cropped{image_path.suffix}"
            else:
                output_path = Path(output_path)
            
            # Save original image as fallback
            save_image(person_final, str(output_path))
            return str(output_path)
        
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

