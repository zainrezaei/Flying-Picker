"""
pipeline.py — Main vision pipeline with live OpenCV overlay.

Reads frames from a video file, detects a single light-coloured
rectangular object, and displays a live window with:
  • Green rotated bounding box
  • Red centroid dot
  • (x, y, θ) text overlay
"""

import os
import sys
import time

import cv2 as cv
import numpy as np
import yaml

from .frame_source import FrameSource
from .preprocess import preprocess
from .detection import detect_object

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load_config(config_path: str | None = None) -> dict:
    """Load the YAML configuration file."""
    if config_path is None:
        config_path = os.path.join(_PROJECT_ROOT, "config", "vision_config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _draw_overlay(frame: np.ndarray, result, cfg_display: dict) -> np.ndarray:
    """Draw bounding box, centroid, and text on the frame (in-place)."""
    overlay_color  = tuple(cfg_display.get("overlay_color", [0, 255, 0]))
    centroid_color = tuple(cfg_display.get("centroid_color", [0, 0, 255]))
    text_color     = tuple(cfg_display.get("text_color", [255, 255, 255]))
    thickness      = cfg_display.get("box_thickness", 2)
    radius         = cfg_display.get("centroid_radius", 6)
    font_scale     = cfg_display.get("font_scale", 0.6)

    # Rotated bounding box
    cv.drawContours(frame, [result.box_points], 0, overlay_color, thickness)

    # Centroid
    cx, cy = int(result.center_x), int(result.center_y)
    cv.circle(frame, (cx, cy), radius, centroid_color, -1)

    # Text label
    label = f"x={cx}  y={cy}  angle={result.angle:.1f} deg"
    cv.putText(
        frame, label,
        (cx + 12, cy - 12),
        cv.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        2,
        cv.LINE_AA,
    )

    return frame


# ------------------------------------------------------------------ #
# Main pipeline                                                        #
# ------------------------------------------------------------------ #

def run_pipeline(config_path: str | None = None):
    """Run the full vision pipeline with live OpenCV display.

    Parameters
    ----------
    config_path : str or None
        Path to YAML config. Defaults to config/vision_config.yaml.
    """
    cfg = _load_config(config_path)

    # --- Unpack config -----------------------------------------------
    video_path = os.path.join(_PROJECT_ROOT, cfg["input"]["video_path"])

    blur_kernel = cfg["preprocess"]["blur_kernel_size"]
    thresh_val  = cfg["preprocess"]["threshold_value"]
    thresh_max  = cfg["preprocess"]["threshold_max"]

    min_area    = cfg["detection"]["min_contour_area"]

    show_mask   = cfg["display"].get("show_mask", True)
    cfg_display = cfg["display"]

    # Region of interest (crop frame edges to remove noise)
    roi_cfg = cfg.get("roi", None)
    use_roi = roi_cfg is not None and roi_cfg.get("enabled", False)

    # --- Open source -------------------------------------------------
    source = FrameSource(video_path, loop=True)
    delay  = max(1, int(1000 / source.fps))

    print(f"[pipeline] Opened {source}")
    print(f"[pipeline] Press 'q' to quit.\n")

    frame_num = 0

    while True:
        frame = source.read()
        if frame is None:
            print("[pipeline] End of source.")
            break

        frame_num += 1
        t0 = time.perf_counter()

        # Apply ROI crop if configured
        roi_x, roi_y = 0, 0
        if use_roi:
            h, w = frame.shape[:2]
            roi_x = int(w * roi_cfg.get("x_start", 0))
            roi_y = int(h * roi_cfg.get("y_start", 0))
            roi_x2 = int(w * roi_cfg.get("x_end", 1))
            roi_y2 = int(h * roi_cfg.get("y_end", 1))
            cropped = frame[roi_y:roi_y2, roi_x:roi_x2]
        else:
            cropped = frame

        # 1. Preprocess
        mask = preprocess(cropped, blur_kernel, thresh_val, thresh_max)

        # 2. Detect
        result = detect_object(mask, min_area)

        # 3. Offset coordinates back to full frame
        if result is not None and use_roi:
            result.center_x += roi_x
            result.center_y += roi_y
            result.box_points[:, 0] += roi_x
            result.box_points[:, 1] += roi_y

        # 3. Overlay & display
        if result is not None:
            _draw_overlay(frame, result, cfg_display)
            dt_ms = (time.perf_counter() - t0) * 1000
            print(
                f"[frame {frame_num:>5d}]  "
                f"x={result.center_x:7.1f}  "
                f"y={result.center_y:7.1f}  "
                f"angle={result.angle:6.1f}°  "
                f"({dt_ms:.1f} ms)"
            )
        else:
            print(f"[frame {frame_num:>5d}]  No object detected")

        cv.imshow("Flying Picker - Detection", frame)

        if show_mask:
            cv.imshow("Flying Picker - Mask", mask)

        # Quit on 'q'
        if cv.waitKey(delay) & 0xFF == ord("q"):
            print("\n[pipeline] Quit by user.")
            break

    source.release()
    cv.destroyAllWindows()


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    # Allow: python -m src.vision.pipeline [config.yaml]
    cfg_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(cfg_arg)
