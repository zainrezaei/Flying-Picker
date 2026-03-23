"""
detection.py — Object detection via contour analysis.

Finds the largest light-coloured object in a binary mask and returns
its centroid (x, y), bounding-box dimensions, rotation angle, and a
confidence score indicating how likely the detection is a real object.
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
    confidence: float = 0.0  # 0.0–1.0 detection confidence


def _compute_confidence(contour: np.ndarray, area: float, w: float, h: float) -> float:
    """Compute a 0–1 confidence score from geometric properties.

    Components (equally weighted):
      1. Solidity   — contour area / convex-hull area  (1.0 = perfectly convex)
      2. Rectangularity — contour area / bounding-box area (1.0 = perfect rectangle)
      3. Area ratio — penalises very small contours that barely pass min_area
    """
    # Solidity
    hull = cv.convexHull(contour)
    hull_area = cv.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    # Rectangularity (how well the contour fills its rotated bounding box)
    bbox_area = w * h if (w > 0 and h > 0) else 1.0
    rectangularity = area / bbox_area

    # Combine (simple average — all components are 0–1)
    confidence = (solidity + rectangularity) / 2.0
    return round(min(max(confidence, 0.0), 1.0), 3)


def detect_object(
    mask: np.ndarray,
    min_area: int = 5000,
    max_area: int = 0,
    min_solidity: float = 0.0,
    min_aspect_ratio: float = 0.0,
    max_aspect_ratio: float = 0.0,
    edge_margin: int = 0,
) -> Optional[DetectionResult]:
    """Detect the single largest object in a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary image (white object on black background).
    min_area : int
        Ignore contours whose area is smaller than this (pixels²).
    max_area : int
        Ignore contours whose area is larger than this (pixels²).
        0 = no upper limit.
    min_solidity : float
        Minimum contour-area / convex-hull-area ratio (0–1).
        0 = disabled.
    min_aspect_ratio : float
        Minimum width/height ratio of the rotated bounding box.
        0 = disabled.
    max_aspect_ratio : float
        Maximum width/height ratio of the rotated bounding box.
        0 = disabled.

    edge_margin : int
        Reject contours whose bounding box is within this many pixels
        of the frame edge (catches partially-visible parts). 0 = disabled.

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
    area = cv.contourArea(largest)

    # --- Gate 1: area bounds ---
    if area < min_area:
        return None
    if max_area > 0 and area > max_area:
        return None

    # --- Gate 1b: edge proximity (reject partially visible objects) ---
    if edge_margin > 0:
        frame_h, frame_w = mask.shape[:2]
        bx, by, bw, bh = cv.boundingRect(largest)
        if (bx <= edge_margin or
            by <= edge_margin or
            bx + bw >= frame_w - edge_margin or
            by + bh >= frame_h - edge_margin):
            return None

    # --- Gate 2: solidity ---
    if min_solidity > 0:
        hull = cv.convexHull(largest)
        hull_area = cv.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0
        if solidity < min_solidity:
            return None

    # Minimum-area rotated rectangle
    rect = cv.minAreaRect(largest)      # ((cx, cy), (w, h), angle)
    (cx, cy), (w, h), angle = rect

    # --- Gate 3: aspect ratio ---
    if w > 0 and h > 0:
        aspect = max(w, h) / min(w, h)
    else:
        aspect = 0.0

    if min_aspect_ratio > 0 and aspect < min_aspect_ratio:
        return None
    if max_aspect_ratio > 0 and aspect > max_aspect_ratio:
        return None

    # Normalise angle so it's easier to interpret:
    # OpenCV's minAreaRect returns angle in [-90, 0).
    # We convert so that 0° = aligned with x-axis, positive = CCW.
    if w < h:
        angle = angle + 90              # swap so width > height convention

    box = cv.boxPoints(rect)            # 4 corner points
    box = np.intp(box)                  # convert to integer

    confidence = _compute_confidence(largest, area, w, h)

    return DetectionResult(
        center_x=cx,
        center_y=cy,
        width=w,
        height=h,
        angle=angle,
        contour=largest,
        box_points=box,
        confidence=confidence,
    )
