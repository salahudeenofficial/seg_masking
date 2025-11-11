"""
OpenPose implementation - optional fallback if dependencies not available.
If OpenPose is not available, returns default keypoints.
"""

from pathlib import Path
import sys
import numpy as np
from PIL import Image

# Try to import OpenPose dependencies
OPENPOSE_AVAILABLE = False
try:
    from pathlib import Path
    import sys
    
    PROJECT_ROOT = Path(__file__).absolute().parents[2].absolute()
    sys.path.insert(0, str(PROJECT_ROOT))
    
    # Try multiple import paths
    try:
        # Try from parent mask_gpu if available
        parent_mask_gpu = PROJECT_ROOT.parent / "mask_gpu" / "StableVITON" / "preprocess" / "openpose"
        if parent_mask_gpu.exists():
            sys.path.insert(0, str(parent_mask_gpu.parent))
            from preprocess.openpose.annotator.util import resize_image, HWC3
            from preprocess.openpose.annotator.openpose import OpenposeDetector
            OPENPOSE_AVAILABLE = True
    except ImportError:
        try:
            # Try direct import
            from annotator.util import resize_image, HWC3
            from annotator.openpose import OpenposeDetector
            OPENPOSE_AVAILABLE = True
        except ImportError:
            pass
except Exception:
    pass

if not OPENPOSE_AVAILABLE:
    print("Warning: OpenPose dependencies not found. Using fallback (default keypoints).")
    print("Note: Masking will still work using SegFormer parsing, but pose detection will be disabled.")


class OpenPose:
    """
    OpenPose wrapper with fallback to default keypoints if dependencies unavailable.
    OpenPose is optional - SegFormer parsing works independently.
    """
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.has_openpose = OPENPOSE_AVAILABLE
        if self.has_openpose:
            try:
                import torch
                self.preprocessor = OpenposeDetector()
            except Exception as e:
                print(f"Warning: Could not initialize OpenPose: {e}")
                self.has_openpose = False
        else:
            self.preprocessor = None

    def __call__(self, input_image, resolution=384):
        """
        Get body keypoints. Returns default keypoints if OpenPose unavailable.
        """
        # If OpenPose not available, return default keypoints
        if not self.has_openpose:
            # Return default keypoints (all zeros) - this is fine for SegFormer-based masking
            candidate = [[0, 0] for _ in range(18)]
            return {"pose_keypoints_2d": candidate}
        
        # Use actual OpenPose if available
        try:
            import torch
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
        except Exception as e:
            # Fallback to default keypoints on any error
            print(f"Warning: OpenPose processing failed: {e}. Using default keypoints.")
            candidate = [[0, 0] for _ in range(18)]
            return {"pose_keypoints_2d": candidate}

