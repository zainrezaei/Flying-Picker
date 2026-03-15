import json
import os
import glob
from dataclasses import dataclass, field
from typing import Optional

import cv2 as cv
import numpy as np

# ------------------------------------------------------------------ #
# Data                                                                 #
# ------------------------------------------------------------------ #

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_CALIB_PATH = os.path.join(_PROJECT_ROOT, "config", "camera_calibration.json")
IMAGE_DIR = os.path.join(_PROJECT_ROOT, "calib_images")

@dataclass
class CalibrationResult:
    """Holds the output of camera calibration."""

    camera_matrix: np.ndarray       # 3×3 intrinsic matrix
    dist_coeffs: np.ndarray         # distortion coefficients [k1, k2, p1, p2, k3]
    image_size: tuple[int, int]     # (width, height)
    rms_error: float                # reprojection error (lower = better, <0.5 is good)

    # Precomputed undistort maps for fast remapping
    map1: Optional[np.ndarray] = field(default=None, repr=False)
    map2: Optional[np.ndarray] = field(default=None, repr=False)


# ------------------------------------------------------------------ #
# Calibrate from images                                                #
# ------------------------------------------------------------------ #

def calibrate_from_images(
    image_paths: Optional[list[str]] = None,
    board_size: tuple[int, int] = (9, 6),
    square_size_mm: float = 25.0,
    show_corners: bool = True,
) -> CalibrationResult:
    if image_paths is None:
        # Hardcode full path to folder with images
        image_dir = "/home/rasp5/flypicker2/Flying-Picker/calib_images"
        image_paths = (
            glob.glob(os.path.join(image_dir, "*.jpg")) +
            glob.glob(os.path.join(image_dir, "*.jpeg")) +
            glob.glob(os.path.join(image_dir, "*.JPG")) +
            glob.glob(os.path.join(image_dir, "*.JPEG"))
        )

    print(f"[calibration] Processing {len(image_paths)} images from {image_dir} "
          f"(board {board_size[0]}×{board_size[1]}, square {square_size_mm} mm)...")
    

image_dir = "/home/rasp5/flypicker2/Flying-Picker/calib_images"
print(f"Checking contents of: {image_dir}")

files = glob.glob(os.path.join(image_dir, "*"))
print(f"Files found: {files}")
    
if __name__ == "__main__":
    # calibrate from saved images
    calib_result = calibrate_from_images(show_corners=True)