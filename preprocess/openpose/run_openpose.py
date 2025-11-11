"""
OpenPose implementation - reused from original mask_gpu.
This is a copy to keep the alternative approach self-contained.
"""

import pdb
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).absolute().parents[2].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import from parent mask_gpu if available
try:
    # Import from original mask_gpu
    parent_mask_gpu = PROJECT_ROOT.parent / "mask_gpu" / "StableVITON" / "preprocess" / "openpose"
    if parent_mask_gpu.exists():
        sys.path.insert(0, str(parent_mask_gpu.parent))
        from preprocess.openpose.annotator.util import resize_image, HWC3
        from preprocess.openpose.annotator.openpose import OpenposeDetector
    else:
        raise ImportError("Original OpenPose not found")
except ImportError:
    # Fallback: try direct import
    try:
        from annotator.util import resize_image, HWC3
        from annotator.openpose import OpenposeDetector
    except ImportError:
        raise RuntimeError("OpenPose dependencies not found. Please ensure mask_gpu is set up correctly.")

import cv2
import einops
import numpy as np
import argparse
from PIL import Image
import torch


class OpenPose:
    def __init__(self, gpu_id: int):
        self.preprocessor = OpenposeDetector()

    def __call__(self, input_image, resolution=384):
        if isinstance(input_image, Image.Image):
            input_image = np.asarray(input_image)
        elif type(input_image) == str:
            input_image = np.asarray(Image.open(input_image))
        else:
            raise ValueError
        with torch.no_grad():
            input_image = HWC3(input_image)
            input_image = resize_image(input_image, resolution)
            H, W, C = input_image.shape
            assert (H == 512 and W == 384), 'Incorrect input image shape'
            pose, detected_map = self.preprocessor(input_image, hand_and_face=False)

            # Handle case where no body is detected
            try:
                bodies = pose.get('bodies', {})
                subset_list = bodies.get('subset', [])
                candidate = bodies.get('candidate', [])
                
                if not subset_list or len(subset_list) == 0:
                    candidate = [[0, 0] for _ in range(18)]
                    keypoints = {"pose_keypoints_2d": candidate}
                    return keypoints
                
                subset = subset_list[0][:18]
                
                if not isinstance(candidate, list):
                    candidate = candidate.tolist() if hasattr(candidate, 'tolist') else list(candidate)
                
                if len(candidate) == 0:
                    candidate = [[0, 0] for _ in range(18)]
                    keypoints = {"pose_keypoints_2d": candidate}
                    return keypoints
            except (KeyError, IndexError, TypeError) as e:
                candidate = [[0, 0] for _ in range(18)]
                keypoints = {"pose_keypoints_2d": candidate}
                return keypoints
            
            # Process subset and candidate with bounds checking
            try:
                for i in range(18):
                    if i >= len(subset):
                        break
                    if subset[i] == -1:
                        if i < len(candidate):
                            candidate.insert(i, [0, 0])
                        else:
                            candidate.append([0, 0])
                        for j in range(i, 18):
                            if j < len(subset) and subset[j] != -1:
                                subset[j] += 1
                    elif subset[i] != i:
                        if i < len(candidate):
                            candidate.pop(i)
                        for j in range(i, 18):
                            if j < len(subset) and subset[j] != -1:
                                subset[j] -= 1
            except (IndexError, ValueError) as e:
                candidate = [[0, 0] for _ in range(18)]
                keypoints = {"pose_keypoints_2d": candidate}
                return keypoints

            candidate = candidate[:18]
            while len(candidate) < 18:
                candidate.append([0, 0])
            candidate = candidate[:18]

            for i in range(18):
                if i < len(candidate):
                    candidate[i][0] *= 384
                    candidate[i][1] *= 512
                else:
                    candidate.append([0, 0])

            keypoints = {"pose_keypoints_2d": candidate}

        return keypoints

