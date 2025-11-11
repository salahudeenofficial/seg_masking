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
sys.path.insert(0, str(project_root / "preprocess" / "openpose"))

# Global cache for preprocessors to avoid re-initialization
_preprocessors_cache = None


def create_rectangle_mask_from_body_parts(model_parse: Image.Image, category: str, width: int, height: int) -> Image.Image:
    """
    Create a rectangle mask covering each body part detected in the parsing result.
    Each body part gets its own rectangle (bounding box), then all rectangles are combined.
    Excludes shoes and feet from masking.
    """
    parse_array = np.array(model_parse.resize((width, height), Image.NEAREST))
    rectangle_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Exclude shoe/feet labels: 9 (left_shoe), 10 (right_shoe)
    excluded_labels = {9, 10}  # Never mask shoes
    
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
    
    # Process each body part label
    for label in body_part_labels:
        part_mask = (parse_array == label).astype(np.uint8)
        if part_mask.sum() == 0:
            continue
        
        # Find bounding box of this body part
        rows, cols = np.where(part_mask > 0)
        if len(rows) == 0 or len(cols) == 0:
            continue
        
        y_min, y_max = int(rows.min()), int(rows.max())
        x_min, x_max = int(cols.min()), int(cols.max())
        
        # Ensure bounding box stays within image bounds
        rect_x_min = max(0, x_min)
        rect_x_max = min(width, x_max + 1)
        rect_y_min = max(0, y_min)
        rect_y_max = min(height, y_max + 1)
        
        # Draw filled rectangle for this body part
        if rect_x_max > rect_x_min and rect_y_max > rect_y_min:
            cv2.rectangle(rectangle_mask, (rect_x_min, rect_y_min), (rect_x_max - 1, rect_y_max - 1), 255, -1)
    
    return Image.fromarray(rectangle_mask, mode='L')


def resize_panel(img: Image.Image, width: int, height: int) -> Image.Image:
    """Resize image to specified dimensions."""
    return img.convert("RGB").resize((width, height))


def init_preprocessors(use_gpu_device: int = 0):
    """Initialize OpenPose and SegFormer Parsing preprocessors."""
    try:
        from preprocess.humanparsing.run_parsing import SegFormerParsing
        from preprocess.openpose.run_openpose import OpenPose
    except Exception as e:
        raise RuntimeError(f"Preprocessors not available: {e}")
    
    openpose = OpenPose(use_gpu_device)
    try:
        if hasattr(openpose, 'preprocessor') and hasattr(openpose.preprocessor, 'body_estimation'):
            openpose.preprocessor.body_estimation.model.to('cuda')
    except Exception:
        # If CUDA not available, we proceed
        pass
    
    parsing = SegFormerParsing(use_gpu_device)
    
    return openpose, parsing


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
                 preserve_resolution: bool = True) -> str:
    """
    Create a masked image based on mask_type using SegFormer B5 for parsing.
    
    Args:
        mask_type: One of 'upper_body', 'lower_body', or 'other'
        imagepath: Path to the input image
        output_path: Optional path for output image. If None, saves next to input with '_masked' suffix
        width: Working resolution width for processing (default: 576). Only used if preserve_resolution=False
        height: Working resolution height for processing (default: 768). Only used if preserve_resolution=False
        device_index: GPU device index (default: 0)
        preserve_resolution: If True, output will match input image resolution. If False, uses width/height (default: True)
    
    Returns:
        Path to the masked image (or original image path if mask_type is 'other')
    
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
    openpose, parsing = get_preprocessors(use_gpu_device=device_index)
    
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
        # Get keypoints and parsing
        try:
            keypoints = openpose(person_r)
        except Exception as e:
            print(f"ERROR: OpenPose failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise
        
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
        
        # Create rectangle mask from body parts at processing resolution
        square_mask_combined = create_rectangle_mask_from_body_parts(
            model_parse=model_parse,
            category=mask_type,
            width=process_width,
            height=process_height,
        )
        
        # Check if mask is empty (no body parts detected)
        mask_array = np.array(square_mask_combined)
        if mask_array.sum() == 0:
            raise ValueError("No body parts detected in parsing result - mask is empty")
        
        # Scale mask back to original resolution if preserving resolution
        if preserve_resolution and (process_width != original_width or process_height != original_height):
            square_mask_combined = square_mask_combined.resize(
                (original_width, original_height), 
                Image.NEAREST  # Use nearest neighbor to preserve mask boundaries
            )
        
        # Use original image at original resolution for final output
        if preserve_resolution:
            person_final = person_img.convert('RGB')
            person_np = np.array(person_final).astype(np.float32) / 255.0
        else:
            person_np = np.array(person_r.convert('RGB')).astype(np.float32) / 255.0
        
        # Rectangle mask from body parts: use this for green visualization
        square_mask_np = np.array(square_mask_combined.convert('L')).astype(np.float32) / 255.0
        square_mask_3 = np.repeat(square_mask_np[:, :, None], 3, axis=2)
        
        # control_image (masked person): green where garment IS (square_mask_3=1.0), person where visible (square_mask_3=0.0)
        green_color = np.array([0.0, 1.0, 0.0])  # RGB green in [0,1] range
        # Keep person where garment is NOT (square_mask_3=0), fill green where garment IS (square_mask_3=1)
        masked_person_np = person_np * (1.0 - square_mask_3) + green_color * square_mask_3
        masked_person = Image.fromarray((masked_person_np * 255).astype(np.uint8))
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_type = type(e).__name__
        print(f"\n{'='*60}")
        print(f"ERROR: Could not create mask")
        print(f"Error type: {error_type}")
        print(f"Error message: {error_msg}")
        print(f"{'='*60}")
        print("Full traceback:")
        traceback.print_exc()
        print(f"{'='*60}\n")
        
        # Re-raise the exception to see the full error
        raise RuntimeError(f"Masking failed: {error_type}: {error_msg}") from e
    
    # Determine output path
    if output_path is None:
        # Save next to original with '_masked' suffix
        output_path = image_path.parent / f"{image_path.stem}_masked{image_path.suffix}"
    else:
        output_path = Path(output_path)
    
    # Save the masked image
    save_image(masked_person, str(output_path))
    
    return str(output_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create masked image using SegFormer B5")
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
    
    args = parser.parse_args()
    
    try:
        result_path = masked_image(
            mask_type=args.mask_type,
            imagepath=args.imagepath,
            output_path=args.output,
            width=args.width,
            height=args.height,
            device_index=args.device,
            preserve_resolution=args.preserve_resolution
        )
        print(f"Masked image saved to: {result_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

