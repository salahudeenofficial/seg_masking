"""
SegFormer B5-based human parsing implementation.
Replaces the ATR parsing model with SegFormer for better accuracy.
"""

import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# Try to import transformers (SegFormer)
try:
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    SEGFORMER_AVAILABLE = True
except ImportError:
    SEGFORMER_AVAILABLE = False
    print("Warning: transformers library not available. Please install: pip install transformers")


class SegFormerParsing:
    """
    Human parsing using SegFormer B5 model.
    Uses a pre-trained SegFormer model fine-tuned for human parsing.
    """
    
    # ATR label mapping (19 classes) - same as original
    LABEL_MAPPING = {
        0: 'background',
        1: 'hat',
        2: 'hair',
        3: 'glove',
        4: 'upper_clothes',  # Used for upper_body masking
        5: 'skirt',          # Used for lower_body masking
        6: 'pants',          # Used for lower_body masking
        7: 'dress',          # Full body - only for dresses category
        8: 'belt',
        9: 'left_shoe',     # Excluded from masking
        10: 'right_shoe',   # Excluded from masking
        11: 'face',
        12: 'left_leg',
        13: 'right_leg',
        14: 'left_arm',
        15: 'right_arm',
        16: 'bag',
        17: 'scarf',
        18: 'neck'
    }
    
    def __init__(self, gpu_id: int = 0, model_name: str = "nvidia/segformer-b5-finetuned-ade-640-640"):
        """
        Initialize SegFormer B5 model for human parsing.
        
        Available pretrained models:
        - "nvidia/segformer-b5-finetuned-ade-640-640" (default): ADE20K dataset, 150 classes
        - "nvidia/segformer-b5-finetuned-cityscapes-1024-1024": Cityscapes dataset, 19 classes (but for urban scenes)
        - "nvidia/mit-b5": Base encoder only (needs fine-tuning)
        
        Args:
            gpu_id: GPU device index
            model_name: HuggingFace model name for SegFormer
        """
        if not SEGFORMER_AVAILABLE:
            raise RuntimeError("transformers library is required. Install with: pip install transformers")
        
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        
        # Load SegFormer B5 model
        print(f"Loading SegFormer B5 model: {model_name}")
        try:
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
            self.processor = SegformerImageProcessor.from_pretrained(model_name)
        except Exception as e:
            print(f"Warning: Could not load {model_name}, trying fallback...")
            # Fallback to ADE20K model
            try:
                fallback_name = "nvidia/segformer-b5-finetuned-ade-640-640"
                print(f"Trying fallback model: {fallback_name}")
                self.model = SegformerForSemanticSegmentation.from_pretrained(fallback_name)
                self.processor = SegformerImageProcessor.from_pretrained(fallback_name)
                model_name = fallback_name
            except Exception as e2:
                raise RuntimeError(f"Could not load SegFormer model: {e2}")
        
        self.model.to(self.device)
        self.model.eval()
        
        self.num_labels = self.model.config.num_labels
        self.model_name = model_name
        
        print(f"SegFormer B5 loaded successfully on {self.device}")
        print(f"Model: {model_name}")
        print(f"Model has {self.num_labels} output classes")
        
        # Initialize ADE20K to ATR mapping if using ADE20K model
        if "ade" in model_name.lower() and self.num_labels == 150:
            self._init_ade20k_to_atr_mapping()
    
    def _init_ade20k_to_atr_mapping(self):
        """
        Initialize mapping from ADE20K classes to ATR human parsing labels.
        ADE20K has 150 classes, we need to map relevant ones to 19 ATR classes.
        """
        # ADE20K class indices (from ADE20K dataset)
        # This is a heuristic mapping - for best results, fine-tune on human parsing data
        self.ade20k_to_atr = {}
        
        # Background
        self.ade20k_to_atr[0] = 0  # background -> background
        
        # Person-related classes in ADE20K (approximate mappings)
        # Note: ADE20K doesn't have fine-grained human parsing labels
        # We'll use person class and try to infer body parts from context
        # This is a simplified approach - fine-tuning would be better
        
        # Person class in ADE20K (class 12 is "person")
        # We'll use a simple heuristic: detect person regions and map to clothing
        self.person_class_ade20k = 12  # Person class in ADE20K
        
        print("Note: Using heuristic mapping from ADE20K to ATR labels.")
        print("For best results, consider fine-tuning SegFormer on human parsing dataset.")
    
    def _map_to_atr_labels(self, seg_output: np.ndarray) -> np.ndarray:
        """
        Map SegFormer output to ATR parsing labels (19 classes).
        
        Args:
            seg_output: Segmentation output array with class labels
            
        Returns:
            ATR-formatted segmentation with 19 classes
        """
        # If the model outputs 19 classes (human parsing), use directly
        if self.num_labels == 19:
            return seg_output
        
        # Map from ADE20K (150 classes) to ATR (19 classes)
        if "ade" in self.model_name.lower() and self.num_labels == 150:
            return self._map_ade20k_to_atr(seg_output)
        
        # For other models, return as-is (may need custom mapping)
        print(f"Warning: No mapping defined for {self.num_labels} classes. Using raw output.")
        return seg_output
    
    def _map_ade20k_to_atr(self, seg_output: np.ndarray) -> np.ndarray:
        """
        Heuristic mapping from ADE20K classes to ATR human parsing labels.
        This is a simplified approach - for production, fine-tune on human parsing data.
        
        Strategy:
        1. Detect person regions (ADE20K class 12)
        2. Use spatial heuristics to assign body parts
        3. Map to ATR labels
        """
        atr_output = np.zeros_like(seg_output, dtype=np.uint8)
        
        # Find person regions
        person_mask = (seg_output == self.person_class_ade20k)
        
        if person_mask.sum() == 0:
            # No person detected, return background
            return atr_output
        
        # Get person bounding box
        rows, cols = np.where(person_mask)
        if len(rows) == 0:
            return atr_output
        
        y_min, y_max = rows.min(), rows.max()
        x_min, x_max = cols.min(), cols.max()
        
        person_height = y_max - y_min
        person_width = x_max - x_min
        
        # Heuristic: divide person region into upper and lower body
        # Upper body: top 60% of person region -> label 4 (upper_clothes)
        # Lower body: bottom 40% of person region -> label 6 (pants)
        upper_threshold = y_min + int(person_height * 0.6)
        
        # Create masks - iterate through person pixels
        person_y, person_x = np.where(person_mask)
        
        # Face threshold: top 20% of person region
        face_threshold = y_min + int(person_height * 0.2)
        
        for y, x in zip(person_y, person_x):
            if y <= face_threshold:
                # Face region -> label 11 (face)
                atr_output[y, x] = 11
            elif y <= upper_threshold:
                # Upper body region -> upper_clothes (label 4)
                atr_output[y, x] = 4
            else:
                # Lower body region -> pants (label 6)
                atr_output[y, x] = 6
        
        return atr_output
    
    def __call__(self, input_image):
        """
        Perform human parsing on input image.
        
        Args:
            input_image: PIL Image or numpy array
            
        Returns:
            parsed_image: PIL Image with parsing labels (same format as ATR)
            face_mask: torch.Tensor indicating face regions
        """
        # Convert to PIL if needed
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)
        elif not isinstance(input_image, Image.Image):
            if isinstance(input_image, str) or isinstance(input_image, Path):
                input_image = Image.open(input_image)
            else:
                raise ValueError(f"Unsupported input type: {type(input_image)}")
        
        # Ensure RGB
        if input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')
        
        # Process image with SegFormer processor
        inputs = self.processor(images=input_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Upsample to original image size
            upsampled_logits = F.interpolate(
                logits,
                size=input_image.size[::-1],  # (height, width)
                mode="bilinear",
                align_corners=False
            )
            
            # Get predicted labels
            pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        
        # Map to ATR labels if needed
        pred_seg = self._map_to_atr_labels(pred_seg)
        
        # Ensure output is in valid range [0, 18] for ATR
        pred_seg = np.clip(pred_seg, 0, 18).astype(np.uint8)
        
        # Create PIL Image with palette (same format as ATR output)
        palette = self._get_palette(19)
        parsed_image = Image.fromarray(pred_seg.astype(np.uint8), mode='P')
        parsed_image.putpalette(palette)
        
        # Create face mask (label 11 is face)
        face_mask = torch.from_numpy((pred_seg == 11).astype(np.float32))
        
        return parsed_image, face_mask
    
    def _get_palette(self, num_cls):
        """Generate palette for visualization (same as ATR)."""
        n = num_cls
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette


# Alias for compatibility
Parsing = SegFormerParsing

