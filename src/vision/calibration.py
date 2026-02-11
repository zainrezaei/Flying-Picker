"""
calibration.py — Camera intrinsic calibration using a checkerboard pattern.

Computes the camera matrix and distortion coefficients needed to
correct lens distortion (barrel/pincushion). Results are saved to
a JSON file and loaded by the pipeline to undistort every frame.

Usage (standalone):
    python -m src.vision.calibration          # interactive capture
    python -m src.vision.calibration --dir calibration_images/

When the camera hardware arrives:
    1. Print a checkerboard (default 9×6 inner corners)
    2. Run this script — it captures or loads images
    3. Results are saved to config/camera_calibration.json
    4. The pipeline automatically loads and applies undistortion
"""

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
    image_paths: list[str],
    board_size: tuple[int, int] = (9, 6),
    square_size_mm: float = 25.0,
    show_corners: bool = True,
) -> CalibrationResult:
    """Run camera calibration from a set of checkerboard images.

    Parameters
    ----------
    image_paths : list[str]
        Paths to checkerboard images (at least 10 recommended).
    board_size : (cols, rows)
        Number of inner corners in the checkerboard.
    square_size_mm : float
        Physical size of one checkerboard square in mm.
    show_corners : bool
        If True, display detected corners for visual verification.

    Returns
    -------
    CalibrationResult
    """
    # Prepare object points (0,0,0), (1,0,0), ... scaled by square size
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm

    obj_points = []   # 3D points in world space
    img_points = []   # 2D points in image space
    image_size = None

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print(f"[calibration] Processing {len(image_paths)} images "
          f"(board {board_size[0]}×{board_size[1]}, square {square_size_mm} mm)...")

    for i, path in enumerate(image_paths):
        img = cv.imread(path)
        if img is None:
            print(f"  [{i+1}] SKIP — cannot read: {path}")
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])  # (w, h)

        found, corners = cv.findChessboardCorners(gray, board_size, None)

        if found:
            # Refine corner positions to sub-pixel accuracy
            corners_refined = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners_refined)
            print(f"  [{i+1}] OK    — {os.path.basename(path)}")

            if show_corners:
                vis = img.copy()
                cv.drawChessboardCorners(vis, board_size, corners_refined, found)
                cv.imshow("Checkerboard Corners", vis)
                cv.waitKey(300)
        else:
            print(f"  [{i+1}] SKIP  — no corners found: {os.path.basename(path)}")

    if show_corners:
        cv.destroyWindow("Checkerboard Corners")

    if len(obj_points) < 3:
        raise ValueError(
            f"Only {len(obj_points)} valid images found. "
            "Need at least 3 (10+ recommended) for reliable calibration."
        )

    print(f"\n[calibration] Running cv2.calibrateCamera() "
          f"with {len(obj_points)} valid images...")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    print(f"[calibration] Done — RMS reprojection error: {rms:.4f}")
    print(f"[calibration] Camera matrix:\n{camera_matrix}")
    print(f"[calibration] Distortion coefficients: {dist_coeffs.ravel()}")

    result = CalibrationResult(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=image_size,
        rms_error=rms,
    )

    # Precompute undistort maps
    _compute_maps(result)

    return result


# ------------------------------------------------------------------ #
# Live capture calibration (for when camera is available)              #
# ------------------------------------------------------------------ #

def calibrate_live(
    source,
    board_size: tuple[int, int] = (9, 6),
    square_size_mm: float = 25.0,
    num_captures: int = 15,
    save_images_dir: str | None = None,
) -> CalibrationResult:
    """Interactive calibration — capture checkerboard images from a live source.

    Parameters
    ----------
    source : FrameSource or any object with .read() method
        The camera/video source.
    board_size : (cols, rows)
        Inner corners of the checkerboard.
    square_size_mm : float
        Size of one square in mm.
    num_captures : int
        How many valid captures to collect.
    save_images_dir : str or None
        If provided, save captured images here for later re-calibration.

    Returns
    -------
    CalibrationResult
    """
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm

    obj_points = []
    img_points = []
    image_size = None

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    if save_images_dir:
        os.makedirs(save_images_dir, exist_ok=True)

    print(f"[calibration] Live capture mode — collecting {num_captures} images")
    print(f"[calibration] Hold the checkerboard in front of the camera.")
    print(f"[calibration] Press SPACE to capture, 'q' to finish early.\n")

    captured = 0

    while captured < num_captures:
        frame = source.read()
        if frame is None:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        # Try to find corners in real-time
        found, corners = cv.findChessboardCorners(
            gray, board_size,
            cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK
        )

        vis = frame.copy()
        if found:
            cv.drawChessboardCorners(vis, board_size, corners, found)

        # Show status
        status = f"Captured: {captured}/{num_captures}"
        if found:
            status += " | CORNERS FOUND — press SPACE to capture"
        else:
            status += " | Move checkerboard..."
        cv.putText(vis, status, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.imshow("Calibration", vis)

        key = cv.waitKey(30) & 0xFF

        if key == ord(" ") and found:
            corners_refined = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners_refined)
            captured += 1
            print(f"  [{captured}/{num_captures}] Captured!")

            if save_images_dir:
                path = os.path.join(save_images_dir, f"calib_{captured:03d}.jpg")
                cv.imwrite(path, frame)

        elif key == ord("q"):
            break

    cv.destroyWindow("Calibration")

    if len(obj_points) < 3:
        raise ValueError(
            f"Only {len(obj_points)} captures. Need at least 3 (10+ recommended)."
        )

    print(f"\n[calibration] Running cv2.calibrateCamera() with {len(obj_points)} images...")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    print(f"[calibration] Done — RMS error: {rms:.4f}")

    result = CalibrationResult(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=image_size,
        rms_error=rms,
    )
    _compute_maps(result)
    return result


# ------------------------------------------------------------------ #
# Save / Load                                                          #
# ------------------------------------------------------------------ #

def save_calibration(result: CalibrationResult, path: str = _DEFAULT_CALIB_PATH):
    """Save calibration to a JSON file."""
    data = {
        "camera_matrix": result.camera_matrix.tolist(),
        "dist_coeffs": result.dist_coeffs.tolist(),
        "image_size": list(result.image_size),
        "rms_error": result.rms_error,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[calibration] Saved to {path}")


def load_calibration(path: str = _DEFAULT_CALIB_PATH) -> CalibrationResult:
    """Load calibration from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    result = CalibrationResult(
        camera_matrix=np.array(data["camera_matrix"]),
        dist_coeffs=np.array(data["dist_coeffs"]),
        image_size=tuple(data["image_size"]),
        rms_error=data["rms_error"],
    )
    _compute_maps(result)
    return result


def calibration_exists(path: str = _DEFAULT_CALIB_PATH) -> bool:
    """Check if a calibration file exists."""
    return os.path.isfile(path)


# ------------------------------------------------------------------ #
# Undistort                                                            #
# ------------------------------------------------------------------ #

def _compute_maps(result: CalibrationResult):
    """Precompute the undistortion remap tables for fast per-frame use."""
    w, h = result.image_size
    new_cam_mtx, roi = cv.getOptimalNewCameraMatrix(
        result.camera_matrix, result.dist_coeffs, (w, h), 1, (w, h)
    )
    result.map1, result.map2 = cv.initUndistortRectifyMap(
        result.camera_matrix, result.dist_coeffs, None, new_cam_mtx, (w, h), cv.CV_16SC2
    )


def undistort_frame(frame: np.ndarray, calib: CalibrationResult) -> np.ndarray:
    """Undistort a single frame using precomputed maps (fast).

    Parameters
    ----------
    frame : np.ndarray
        Raw BGR frame from camera.
    calib : CalibrationResult
        Loaded calibration with precomputed maps.

    Returns
    -------
    np.ndarray
        Undistorted frame.
    """
    if calib.map1 is None or calib.map2 is None:
        _compute_maps(calib)
    return cv.remap(frame, calib.map1, calib.map2, cv.INTER_LINEAR)
