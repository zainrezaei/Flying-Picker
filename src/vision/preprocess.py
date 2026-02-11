"""
preprocess.py — Frame preprocessing for object detection.

Converts a BGR frame into a clean binary mask where the light object
is white (255) and the dark background is black (0).
"""

import cv2 as cv
import numpy as np


def preprocess(
    frame: np.ndarray,
    blur_kernel: int = 5,
    thresh_value: int = 150,
    thresh_max: int = 255,
) -> np.ndarray:
    """Convert a BGR frame to a binary mask.

    Pipeline:
        1. BGR → Grayscale
        2. Gaussian blur (reduce noise)
        3. Binary threshold (light object → white, dark bg → black)
        4. Morphological close (fill small holes in the object)

    Parameters
    ----------
    frame : np.ndarray
        Input BGR image.
    blur_kernel : int
        Gaussian blur kernel size (must be odd).
    thresh_value : int
        Pixel intensity threshold (0-255).
    thresh_max : int
        Value assigned to pixels above threshold.

    Returns
    -------
    np.ndarray
        Binary mask (single-channel, dtype uint8).
    """
    # 1. Grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 2. Gaussian blur
    kernel = (blur_kernel, blur_kernel)
    blurred = cv.GaussianBlur(gray, kernel, 0)

    # 3. Binary threshold
    _, mask = cv.threshold(blurred, thresh_value, thresh_max, cv.THRESH_BINARY)

    # 4. Morphological close — fill small gaps inside the object
    morph_kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, morph_kernel)

    return mask
