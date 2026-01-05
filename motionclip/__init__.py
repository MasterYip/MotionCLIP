"""
MotionCLIP: Text-Driven Human Motion Generation

This package provides tools for text-to-motion and motion-to-text conversion
using CLIP and motion generation models.
"""

import sys
import os

# Add src directory to path for imports
_package_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_package_dir)
_src_dir = os.path.join(_root_dir, 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from src.utils.get_model_and_data import get_motion_clip, get_model_and_data
import src.utils.rotation_conversions as geometry


__version__ = '1.0.0'
__all__ = ['get_motion_clip', 'get_model_and_data', 'geometry']
