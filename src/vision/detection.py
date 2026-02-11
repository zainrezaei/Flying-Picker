"""
detection.py — Object detection via contour analysis.

Finds the largest light-coloured object in a binary mask and returns
its centroid (x, y), bounding-box dimensions, and rotation angle.
"""

from dataclasses import dataclass
from typing import Optional

import cv2 as cv
import numpy as np


@dataclass
class DetectionResult:
    """Holds the detection output for a single object."""

    center_x: float          # centroid x (pixels)
    center_y: float          # centroid y (pixels)
    width: float             # bounding-box width (pixels)
    height: float            # bounding-box height (pixels)
    angle: float             # rotation angle (degrees, −90 to 0)
    contour: np.ndarray      # the raw contour points
    box_points: np.ndarray   # 4 corners of the rotated bounding box


def detect_object(
    mask: np.ndarray,
    min_area: int = 5000,
) -> Optional[DetectionResult]:
    """Detect the single largest object in a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary image (white object on black background).
    min_area : int
        Ignore contours whose area is smaller than this.

    Returns
    -------
    DetectionResult or None
        Detection data, or None if no valid contour was found.
    """
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Pick the largest contour by area
    largest = max(contours, key=cv.contourArea)

    if cv.contourArea(largest) < min_area:
        return None

    # Minimum-area rotated rectangle
    rect = cv.minAreaRect(largest)      # ((cx, cy), (w, h), angle)
    (cx, cy), (w, h), angle = rect

    # Normalise angle so it's easier to interpret:
    # OpenCV's minAreaRect returns angle in [-90, 0).
    # We convert so that 0° = aligned with x-axis, positive = CCW.
    if w < h:
        angle = angle + 90              # swap so width > height convention

    box = cv.boxPoints(rect)            # 4 corner points
    box = np.intp(box)                  # convert to integer

    return DetectionResult(
        center_x=cx,
        center_y=cy,
        width=w,
        height=h,
        angle=angle,
        contour=largest,
        box_points=box,
    )
