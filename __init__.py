"""
SegFormer-based masking module.
Alternative to mask_gpu using SegFormer B5 for human parsing.
"""

from .mask import masked_image, get_preprocessors

__all__ = ['masked_image', 'get_preprocessors']

