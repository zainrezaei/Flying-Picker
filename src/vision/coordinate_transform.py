"""
coordinate_transform.py — Pixel ↔ world coordinate mapping + belt compensation.

Provides:
  1. Homography calibration (pixel points → world mm points)
  2. pixel_to_world()  — convert detection (x_px, y_px) → (X_mm, Y_mm)
  3. compensate_belt_motion() — predict pick position from belt speed + delay

The homography is computed once during calibration and saved to
config/homography.json. The pipeline loads it and applies the
transform to every detection.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

import cv2 as cv
import numpy as np

# ------------------------------------------------------------------ #
# Paths                                                                #
# ------------------------------------------------------------------ #

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_HOMOGRAPHY_PATH = os.path.join(_PROJECT_ROOT, "config", "homography.json")


# ------------------------------------------------------------------ #
# Data                                                                 #
# ------------------------------------------------------------------ #

@dataclass
class WorldCoordinate:
    """Object pose in real-world conveyor coordinates."""

    x_mm: float         # mm from left edge of belt (or robot-origin X)
    y_mm: float         # mm along belt direction (or robot-origin Y)
    angle_deg: float    # rotation in degrees (unchanged from pixel detection)


@dataclass
class HomographyData:
    """Holds the homography matrix and the calibration points used."""

    matrix: np.ndarray                # 3×3 homography matrix
    pixel_points: np.ndarray          # Nx2 pixel calibration points
    world_points: np.ndarray          # Nx2 world calibration points (mm)
    reprojection_error: float         # average error in mm


# ------------------------------------------------------------------ #
# Homography calibration                                               #
# ------------------------------------------------------------------ #

def calibrate_homography(
    pixel_points: np.ndarray,
    world_points: np.ndarray,
) -> HomographyData:
    """Compute the homography from pixel ↔ world point correspondences.

    Parameters
    ----------
    pixel_points : np.ndarray, shape (N, 2)
        Points in pixel coordinates (from clicking on the camera image).
    world_points : np.ndarray, shape (N, 2)
        Corresponding points in world coordinates (mm on the conveyor).
        Must be same length as pixel_points. N >= 4.

    Returns
    -------
    HomographyData

    Example
    -------
    >>> pixel_pts = np.array([[100, 50], [500, 50], [500, 400], [100, 400]], dtype=np.float32)
    >>> world_pts = np.array([[0, 0], [400, 0], [400, 300], [0, 300]], dtype=np.float32)  # mm
    >>> h = calibrate_homography(pixel_pts, world_pts)
    >>> print(h.matrix)
    """
    pixel_points = np.array(pixel_points, dtype=np.float32)
    world_points = np.array(world_points, dtype=np.float32)

    if len(pixel_points) < 4:
        raise ValueError(f"Need at least 4 point pairs, got {len(pixel_points)}")

    if len(pixel_points) != len(world_points):
        raise ValueError("pixel_points and world_points must have the same length")

    H, mask = cv.findHomography(pixel_points, world_points, cv.RANSAC, 5.0)

    if H is None:
        raise ValueError("Homography computation failed — check your point pairs")

    # Compute reprojection error
    transformed = cv.perspectiveTransform(
        pixel_points.reshape(-1, 1, 2), H
    ).reshape(-1, 2)
    errors = np.linalg.norm(transformed - world_points, axis=1)
    avg_error = float(np.mean(errors))

    print(f"[homography] Computed from {len(pixel_points)} point pairs")
    print(f"[homography] Average reprojection error: {avg_error:.2f} mm")
    print(f"[homography] Matrix:\n{H}")

    return HomographyData(
        matrix=H,
        pixel_points=pixel_points,
        world_points=world_points,
        reprojection_error=avg_error,
    )


def calibrate_homography_interactive(
    source,
    world_points: np.ndarray,
) -> HomographyData:
    """Interactive homography calibration — click points on a live/frozen frame.

    Displays a frame and lets the user click on reference points
    whose real-world positions (in mm) are known.

    Parameters
    ----------
    source : FrameSource or similar
        Frame source (video file or camera).
    world_points : np.ndarray, shape (N, 2)
        The real-world (mm) positions of the reference points, in the
        order the user will click them.

    Returns
    -------
    HomographyData
    """
    frame = source.read()
    if frame is None:
        raise RuntimeError("Cannot read a frame from the source")

    pixel_points = []
    n_points = len(world_points)

    def _on_click(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN and len(pixel_points) < n_points:
            pixel_points.append([x, y])
            print(f"  Point {len(pixel_points)}/{n_points}: pixel ({x}, {y}) "
                  f"→ world ({world_points[len(pixel_points)-1][0]}, "
                  f"{world_points[len(pixel_points)-1][1]}) mm")

    win_name = "Homography Calibration — Click reference points"
    cv.namedWindow(win_name)
    cv.setMouseCallback(win_name, _on_click)

    print(f"\n[homography] Click {n_points} reference points on the image.")
    print(f"[homography] World positions (mm): {world_points.tolist()}")
    print(f"[homography] Press 'q' to cancel.\n")

    while len(pixel_points) < n_points:
        vis = frame.copy()

        # Draw already-clicked points
        for i, pt in enumerate(pixel_points):
            cv.circle(vis, tuple(pt), 8, (0, 0, 255), -1)
            label = f"P{i+1} ({world_points[i][0]}, {world_points[i][1]}) mm"
            cv.putText(vis, label, (pt[0] + 12, pt[1] - 8),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)

        # Show instruction
        remaining = n_points - len(pixel_points)
        cv.putText(vis, f"Click point {len(pixel_points)+1}/{n_points} — {remaining} remaining",
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)

        cv.imshow(win_name, vis)
        if cv.waitKey(30) & 0xFF == ord("q"):
            cv.destroyWindow(win_name)
            raise RuntimeError("Calibration cancelled by user")

    cv.destroyWindow(win_name)

    pixel_pts = np.array(pixel_points, dtype=np.float32)
    return calibrate_homography(pixel_pts, world_points)


# ------------------------------------------------------------------ #
# Transform                                                            #
# ------------------------------------------------------------------ #

def pixel_to_world(
    x_px: float,
    y_px: float,
    angle_deg: float,
    homography: HomographyData,
) -> WorldCoordinate:
    """Convert a pixel-space detection to world coordinates (mm).

    Parameters
    ----------
    x_px, y_px : float
        Centroid in pixels.
    angle_deg : float
        Rotation angle in degrees (passed through unchanged).
    homography : HomographyData
        The calibrated homography.

    Returns
    -------
    WorldCoordinate
    """
    pt = np.array([[[x_px, y_px]]], dtype=np.float32)
    transformed = cv.perspectiveTransform(pt, homography.matrix)
    x_mm, y_mm = transformed[0, 0]

    return WorldCoordinate(
        x_mm=float(x_mm),
        y_mm=float(y_mm),
        angle_deg=angle_deg,
    )


# ------------------------------------------------------------------ #
# Belt compensation                                                    #
# ------------------------------------------------------------------ #

def compensate_belt_motion(
    coord: WorldCoordinate,
    belt_speed_mm_s: float,
    delay_s: float,
    belt_direction: str = "y",
) -> WorldCoordinate:
    """Predict where the object will be at pick time.

    Parameters
    ----------
    coord : WorldCoordinate
        Detected world position at time of detection.
    belt_speed_mm_s : float
        Conveyor belt speed in mm/s.
    delay_s : float
        Time from detection to robot reaching the object (seconds).
    belt_direction : str
        Which axis the belt moves along: "x" or "y".

    Returns
    -------
    WorldCoordinate
        Predicted pick position.

    Example
    -------
    >>> coord = WorldCoordinate(x_mm=87.5, y_mm=210.0, angle_deg=14.1)
    >>> pick = compensate_belt_motion(coord, belt_speed_mm_s=200, delay_s=0.05)
    >>> print(pick.y_mm)   # 210.0 + 200 * 0.05 = 220.0
    """
    offset = belt_speed_mm_s * delay_s

    if belt_direction == "y":
        return WorldCoordinate(
            x_mm=coord.x_mm,
            y_mm=coord.y_mm + offset,
            angle_deg=coord.angle_deg,
        )
    elif belt_direction == "x":
        return WorldCoordinate(
            x_mm=coord.x_mm + offset,
            y_mm=coord.y_mm,
            angle_deg=coord.angle_deg,
        )
    else:
        raise ValueError(f"belt_direction must be 'x' or 'y', got '{belt_direction}'")


# ------------------------------------------------------------------ #
# Save / Load                                                          #
# ------------------------------------------------------------------ #

def save_homography(data: HomographyData, path: str = _DEFAULT_HOMOGRAPHY_PATH):
    """Save homography to a JSON file."""
    payload = {
        "matrix": data.matrix.tolist(),
        "pixel_points": data.pixel_points.tolist(),
        "world_points": data.world_points.tolist(),
        "reprojection_error": data.reprojection_error,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[homography] Saved to {path}")


def load_homography(path: str = _DEFAULT_HOMOGRAPHY_PATH) -> HomographyData:
    """Load homography from a JSON file."""
    with open(path, "r") as f:
        payload = json.load(f)

    return HomographyData(
        matrix=np.array(payload["matrix"], dtype=np.float64),
        pixel_points=np.array(payload["pixel_points"], dtype=np.float32),
        world_points=np.array(payload["world_points"], dtype=np.float32),
        reprojection_error=payload["reprojection_error"],
    )


def homography_exists(path: str = _DEFAULT_HOMOGRAPHY_PATH) -> bool:
    """Check if a homography file exists."""
    return os.path.isfile(path)
