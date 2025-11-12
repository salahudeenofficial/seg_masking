#!/usr/bin/env python3
"""
Batch masking script for entire dataset using SegFormer B5.
Processes all images in combined/men and combined/women folders.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
from mask import masked_image

def mask_dataset(gender='all', skip_existing=True, debug=False):
    """
    Mask all images in the dataset.
    
    Args:
        gender: 'men', 'women', or 'all'
        skip_existing: Skip images that already have masked versions
        debug: Enable debug mode for detailed output
    """
    base_dir = Path(__file__).parent.parent  # Go up to sample_dataset
    results = {
        'men': {'total': 0, 'successful': 0, 'failed': 0, 'skipped': 0},
        'women': {'total': 0, 'successful': 0, 'failed': 0, 'skipped': 0}
    }
    
    genders_to_process = []
    if gender in ['all', 'men']:
        genders_to_process.append('men')
    if gender in ['all', 'women']:
        genders_to_process.append('women')
    
    for g in genders_to_process:
        print(f"\n{'='*60}")
        print(f"Processing {g.upper()} images...")
        print(f"{'='*60}")
        
        csv_path = base_dir / 'combined' / g / 'combined.csv'
        first_images_dir = base_dir / 'combined' / g / 'first_images'
        masked_dir = base_dir / 'combined' / g / 'masked'
        
        # Create masked directory if it doesn't exist
        masked_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if CSV exists
        if not csv_path.exists():
            print(f"Warning: CSV not found at {csv_path}")
            continue
        
        # Check if first_images directory exists
        if not first_images_dir.exists():
            print(f"Warning: first_images directory not found at {first_images_dir}")
            continue
        
        # Read CSV
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            continue
        
        # Filter out rows without mask_type
        df = df[df['mask_type'].notna()]
        df = df[df['mask_type'] != '']
        
        total = len(df)
        results[g]['total'] = total
        
        print(f"Total images to process: {total}")
        
        if total == 0:
            print(f"No images to process for {g}")
            continue
        
        # Process each image
        successful = 0
        failed = 0
        skipped = 0
        
        for idx, row in tqdm(df.iterrows(), total=total, desc=f"Processing {g}"):
            try:
                # Get image filename
                image_filename = row.get('image_filename', '')
                if pd.isna(image_filename) or image_filename == '':
                    # Try first_image_filename (common in combined CSVs)
                    image_filename = row.get('first_image_filename', '')
                if pd.isna(image_filename) or image_filename == '':
                    # Try to get from other columns
                    image_filename = row.get('filename', '')
                    if pd.isna(image_filename) or image_filename == '':
                        # Try product_id
                        product_id = str(row.get('product_id', ''))
                        if product_id:
                            # Look for image with product_id in name
                            possible_files = list(first_images_dir.glob(f"{product_id}*.jpg"))
                            if possible_files:
                                image_filename = possible_files[0].name
                            else:
                                print(f"\n[{idx+1}/{total}] Skipping: No image filename found for product_id {product_id}")
                                failed += 1
                                continue
                        else:
                            print(f"\n[{idx+1}/{total}] Skipping: No image identifier found")
                            failed += 1
                            continue
                
                # Ensure .jpg extension
                if not image_filename.lower().endswith(('.jpg', '.jpeg')):
                    image_filename += '.jpg'
                
                image_path = first_images_dir / image_filename
                
                # Check if image exists
                if not image_path.exists():
                    print(f"\n[{idx+1}/{total}] {image_filename}: Image not found")
                    failed += 1
                    continue
                
                # Get mask_type
                mask_type = str(row.get('mask_type', 'other')).strip().lower()
                if mask_type not in ['upper_body', 'lower_body', 'other']:
                    print(f"\n[{idx+1}/{total}] {image_filename}: Invalid mask_type '{mask_type}', skipping")
                    failed += 1
                    continue
                
                # Skip 'other' type (no masking needed)
                if mask_type == 'other':
                    # Copy original to masked folder for consistency
                    output_path = masked_dir / image_filename
                    if not output_path.exists():
                        import shutil
                        shutil.copy2(image_path, output_path)
                    skipped += 1
                    continue
                
                # Check if already exists
                output_path = masked_dir / image_filename
                if skip_existing and output_path.exists():
                    skipped += 1
                    continue
                
                # Perform masking
                try:
                    result_path = masked_image(
                        mask_type=mask_type,
                        imagepath=str(image_path),
                        output_path=str(output_path),
                        debug=debug
                    )
                    successful += 1
                except Exception as e:
                    error_msg = str(e)
                    print(f"\n[{idx+1}/{total}] {image_filename}: Failed - {error_msg}")
                    failed += 1
                    
                    # If masking failed, copy original image as fallback
                    if not output_path.exists():
                        import shutil
                        try:
                            shutil.copy2(image_path, output_path)
                            print(f"  Copied original image as fallback")
                        except:
                            pass
                    
            except Exception as e:
                print(f"\n[{idx+1}/{total}] Error processing row: {e}")
                failed += 1
        
        results[g]['successful'] = successful
        results[g]['failed'] = failed
        results[g]['skipped'] = skipped
        
        print(f"\n=== {g.upper()} Summary ===")
        print(f"Total: {total}")
        print(f"Successful: {successful}")
        print(f"Skipped (already exists): {skipped}")
        print(f"Failed: {failed}")
    
    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    total_all = sum(r['total'] for r in results.values())
    successful_all = sum(r['successful'] for r in results.values())
    failed_all = sum(r['failed'] for r in results.values())
    skipped_all = sum(r['skipped'] for r in results.values())
    
    print(f"Total: {total_all}")
    print(f"Successful: {successful_all}")
    print(f"Failed: {failed_all}")
    print(f"Skipped: {skipped_all}")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch mask entire dataset using SegFormer")
    parser.add_argument("--gender", choices=['men', 'women', 'all'], default='all',
                       help="Which gender to process (default: all)")
    parser.add_argument("--no-skip", dest="skip_existing", action="store_false",
                       help="Re-process images even if masked version exists")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode (shows detailed parsing info)")
    
    args = parser.parse_args()
    
    try:
        results = mask_dataset(
            gender=args.gender,
            skip_existing=args.skip_existing,
            debug=args.debug
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

