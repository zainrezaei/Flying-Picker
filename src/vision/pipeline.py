"""
pipeline.py — Main vision pipeline with live OpenCV overlay.

Full processing chain:
  1. Capture frame (video file or Pi camera)
  2. Undistort (if camera calibration exists)
  3. ROI crop
  4. Preprocess → binary mask
  5. Detect object → pixel (cx, cy, angle)
  6. Pixel → world mm (if homography calibration exists)
  7. Belt motion compensation (if belt config present)
  8. Draw overlay & print results
"""

import os
import sys
import time
import socket

import cv2 as cv
import numpy as np
import yaml

from .frame_source import FrameSource
from .preprocess import preprocess
from .detection import detect_object
from .calibration import (
    calibration_exists,
    load_calibration,
    undistort_frame,
    CalibrationResult,
)
from .coordinate_transform import (
    homography_exists,
    load_homography,
    pixel_to_world,
    compensate_belt_motion,
    apply_camera_offset,
    pixel_size_to_world,
    HomographyData,
    WorldCoordinate,
)
from .RTDEsender import Sender
from .detection_tracker import DetectionTracker

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


def _draw_overlay(
    frame: np.ndarray,
    result,
    cfg_display: dict,
    world_coord: WorldCoordinate | None = None,
    pick_coord: WorldCoordinate | None = None,
) -> np.ndarray:
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

    # Text label — show world mm if available, else pixel
    if pick_coord is not None:
        label = (
            f"pick=({pick_coord.x_mm:.1f}, {pick_coord.y_mm:.1f}) mm  "
            f"angle={pick_coord.angle_deg:.1f} deg"
        )
    elif world_coord is not None:
        label = (
            f"({world_coord.x_mm:.1f}, {world_coord.y_mm:.1f}) mm  "
            f"angle={world_coord.angle_deg:.1f} deg"
        )
    else:
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

    The pipeline automatically loads calibration files if they exist:
      - config/camera_calibration.json  → lens undistortion
      - config/homography.json          → pixel → world mm

    Parameters
    ----------
    config_path : str or None
        Path to YAML config. Defaults to config/vision_config.yaml.
    """
    cfg = _load_config(config_path)

    ROBOT_IP = "192.168.10.2"
    ROBOT_PORT = 30004
    config_file = "rtde_config.xml"

    rtde_sender = Sender(ROBOT_IP, ROBOT_PORT, config_file)
    attempts = 0
    max_attempts = 5
    connected = False
    while attempts < max_attempts:
        try:
            print("Connecting to robot...")
            rtde_sender.connect()
            print("Connected to robot!")
            connected = True
            break
        except Exception as e:
            attempts += 1
            print(f"Error connecting to robot (attempt {attempts}/5): {e}")
            time.sleep(1)
    if not connected:
        print("Failed to connect to robot after multiple attempts. Exiting.")

    # --- Detection tracker (single-send with debounce) ---------------
    track_cfg = cfg.get("tracking", {})
    tracker = DetectionTracker(
        confirm_frames=track_cfg.get("confirm_frames", 3),
        exit_frames=track_cfg.get("exit_frames", 5),
        distance_threshold_mm=track_cfg.get("distance_threshold_mm", 20.0),
    )
    print(f"[pipeline] Detection tracker: confirm={tracker.confirm_frames} frames, "
          f"exit={tracker.exit_frames} frames, "
          f"distance_threshold={tracker.distance_threshold_mm:.1f} mm")

    # --- Unpack config -----------------------------------------------
    video_path = os.path.join(_PROJECT_ROOT, cfg["input"]["video_path"])

    blur_kernel = cfg["preprocess"]["blur_kernel_size"]
    thresh_val  = cfg["preprocess"]["threshold_value"]
    thresh_max  = cfg["preprocess"]["threshold_max"]

    det_cfg     = cfg["detection"]
    min_area    = det_cfg["min_contour_area"]
    max_area    = det_cfg.get("max_contour_area", 0)
    min_solidity    = det_cfg.get("min_solidity", 0.0)
    min_aspect      = det_cfg.get("min_aspect_ratio", 0.0)
    max_aspect      = det_cfg.get("max_aspect_ratio", 0.0)
    min_confidence  = det_cfg.get("min_confidence", 0.0)
    edge_margin     = det_cfg.get("edge_margin", 0)

    show_mask   = cfg["display"].get("show_mask", True)
    cfg_display = cfg["display"]

    # Region of interest (crop frame edges to remove noise)
    roi_cfg = cfg.get("roi", None)
    use_roi = roi_cfg is not None and roi_cfg.get("enabled", False)

    # Belt compensation config
    belt_cfg = cfg.get("belt", None)
    belt_speed = 0.0
    belt_delay = 0.0
    belt_direction = "x"
    if belt_cfg is not None and belt_cfg.get("enabled", False):
        belt_speed     = belt_cfg.get("speed_mm_s", 0.0)
        belt_delay     = belt_cfg.get("detection_to_pick_delay_s", 0.0)
        belt_direction = belt_cfg.get("direction", "x")

    # Physical measurements from calibration section
    calib_cfg = cfg.get("calibration", {})
    belt_width_mm       = calib_cfg.get("belt_width_mm", 0.0)
    camera_height_mm    = calib_cfg.get("camera_height_mm", 0.0)
    cam_offset_x        = calib_cfg.get("camera_offset_x_mm", 0.0)
    cam_offset_y        = calib_cfg.get("camera_offset_y_mm", 0.0)
    exp_obj_w           = calib_cfg.get("expected_object_width_mm", 0.0)
    exp_obj_h           = calib_cfg.get("expected_object_height_mm", 0.0)
    obj_tol_pct         = calib_cfg.get("object_size_tolerance_pct", 25.0)

    use_cam_offset   = cam_offset_x != 0.0 or cam_offset_y != 0.0
    use_size_check   = exp_obj_w > 0.0 and exp_obj_h > 0.0

    if belt_width_mm > 0:
        print(f"[pipeline] Belt width: {belt_width_mm:.0f} mm")
    if camera_height_mm > 0:
        print(f"[pipeline] Camera height above belt: {camera_height_mm:.0f} mm")
    if use_cam_offset:
        print(f"[pipeline] Camera→robot offset: "
              f"ΔX={cam_offset_x:.1f} mm, ΔY={cam_offset_y:.1f} mm")
    if use_size_check:
        print(f"[pipeline] Expected object size: "
              f"{exp_obj_w:.0f}×{exp_obj_h:.0f} mm (±{obj_tol_pct:.0f}%)")

    # --- Load calibration files (optional) ---------------------------
    calib_path = os.path.join(_PROJECT_ROOT, "config", "camera_calibration.json")
    calib: CalibrationResult | None = None
    if calibration_exists(calib_path):
        calib = load_calibration(calib_path)
        print(f"[pipeline] Camera calibration loaded (RMS={calib.rms_error:.4f})")
    else:
        print("[pipeline] No camera calibration found - skipping undistortion.")

    homog_path = os.path.join(_PROJECT_ROOT, "config", "homography.json")
    homog: HomographyData | None = None
    if homography_exists(homog_path):
        homog = load_homography(homog_path)
        print(f"[pipeline] Homography loaded (err={homog.reprojection_error:.2f} mm)")
    else:
        print("[pipeline] No homography found - output will be in pixels.")

    use_belt = belt_speed > 0 and belt_delay > 0 and homog is not None

    # --- Open source -------------------------------------------------
    source = FrameSource(video_path, loop=True)
    delay  = max(1, int(1000 / source.fps))

    print(f"[pipeline] Opened {source}")
    print(f"[pipeline] Press 'q' to quit.\n")

    object_present = False

    frame_num = 0

    while True:
        frame = source.read()
        if frame is None:
            print("[pipeline] End of source.")
            break

        frame_num += 1
        t0 = time.perf_counter()

        # Step 1: Undistort (if calibrated)
        if calib is not None:
            frame = undistort_frame(frame, calib)

        # Step 2: Apply ROI crop
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

        # Step 3: Preprocess → binary mask
        mask = preprocess(cropped, blur_kernel, thresh_val, thresh_max)

        # Step 4: Detect object
        result = detect_object(
            mask,
            min_area=min_area,
            max_area=max_area,
            min_solidity=min_solidity,
            min_aspect_ratio=min_aspect,
            max_aspect_ratio=max_aspect,
            edge_margin=edge_margin,
        )

        # Gate on confidence
        if result is not None and result.confidence < min_confidence:
            result = None

        # Step 5: Offset coordinates back to full frame
        if result is not None and use_roi:
            result.center_x += roi_x
            result.center_y += roi_y
            result.box_points[:, 0] += roi_x
            result.box_points[:, 1] += roi_y

        # Step 6: Pixel → world mm
        world_coord: WorldCoordinate | None = None
        pick_coord: WorldCoordinate | None = None

        if result is not None and homog is not None:
            world_coord = pixel_to_world(
                result.center_x, result.center_y, result.angle, homog
            )

            # Step 6b: Camera → robot offset
            if use_cam_offset:
                world_coord = apply_camera_offset(
                    world_coord, cam_offset_x, cam_offset_y
                )

            # Step 6c: Object size sanity check
            if use_size_check:
                det_w, det_h = pixel_size_to_world(
                    result.width, result.height,
                    result.center_x, result.center_y, homog
                )
                w_err = abs(det_w - exp_obj_w) / exp_obj_w * 100
                h_err = abs(det_h - exp_obj_h) / exp_obj_h * 100
                if w_err > obj_tol_pct or h_err > obj_tol_pct:
                    print(
                        f"[WARNING] Object size {det_w:.1f}×{det_h:.1f} mm "
                        f"deviates from expected {exp_obj_w:.0f}×{exp_obj_h:.0f} mm "
                        f"(w_err={w_err:.1f}%, h_err={h_err:.1f}%)"
                    )

            # Step 7: Belt compensation → predicted pick position
            if use_belt:
                pick_coord = compensate_belt_motion(
                    world_coord, belt_speed, belt_delay, belt_direction
                )

        # Step 7b: Feed tracker and send to robot (runs every frame)
        coord = pick_coord or world_coord  # None when no detection
        tracker_result = tracker.update(coord)

        if connected:
            try:
                if tracker_result.should_send and tracker_result.coord is not None:
                    c = tracker_result.coord
                    rtde_sender.send_pose(c.x_mm, c.y_mm, c.angle_deg, 1.0)
                    object_present = True
                    print(f"[SEND] Pose sent to robot: x={c.x_mm:.1f} mm, "
                          f"y={c.y_mm:.1f} mm, angle={c.angle_deg:.1f} deg")
                elif object_present and coord is None and tracker.state == "IDLE":
                    rtde_sender.send_pose(0.2, 1.0, 0.0, 0)  # Indicate no object
                    object_present = False
                    print(f"[SEND] No-object signal sent to robot.")
            except Exception as e:
                print(f"[ERROR] Failed to send pose to robot: {e}")
                connected = False

        if not connected and frame_num % 10 == 0:
            try:
                rtde_sender.connect()
                connected = True
                print("[INFO] Reconnected to robot.")
            except Exception as e:
                print(f"[WARNING] Still cannot reconnect to robot: {e}")
            

        # Step 8: Overlay & display
        if result is not None:
            _draw_overlay(frame, result, cfg_display, world_coord, pick_coord)
            dt_ms = (time.perf_counter() - t0) * 1000

            if pick_coord is not None:
                print(
                    f"[frame {frame_num:>5d}]  "
                    f"px=({result.center_x:.0f},{result.center_y:.0f})  "
                    f"world=({world_coord.x_mm:.1f},{world_coord.y_mm:.1f}) mm  "
                    f"pick=({pick_coord.x_mm:.1f},{pick_coord.y_mm:.1f}) mm  "
                    f"angle={pick_coord.angle_deg:.1f} deg  "
                    f"conf={result.confidence:.2f}  "
                    f"({dt_ms:.1f} ms)"
                )
            elif world_coord is not None:
                print(
                    f"[frame {frame_num:>5d}]  "
                    f"px=({result.center_x:.0f},{result.center_y:.0f})  "
                    f"world=({world_coord.x_mm:.1f},{world_coord.y_mm:.1f}) mm  "
                    f"angle={world_coord.angle_deg:.1f} deg  "
                    f"conf={result.confidence:.2f}  "
                    f"({dt_ms:.1f} ms)"
                )
            else:
                print(
                    f"[frame {frame_num:>5d}]  "
                    f"x={result.center_x:7.1f}  "
                    f"y={result.center_y:7.1f}  "
                    f"angle={result.angle:6.1f} deg  "
                    f"conf={result.confidence:.2f}  "
                    f"({dt_ms:.1f} ms)"
                )
        else:
            print(f"[frame {frame_num:>5d}]  No object detected")

        cv.imshow("Flying Picker - Detection", frame)

        if show_mask:

           #resized_mask = cv.resize(mask, (640, 480))
           #cv.imshow("Flying Picker - Mask", resized_mask)
            cv.imshow("Flying Picker - Mask", mask)

        # Quit on 'q'
        if cv.waitKey(delay) & 0xFF == ord("q"):
            print("\n[pipeline] Quit by user.")
            break

    source.release()
    rtde_sender.close()

    cv.destroyAllWindows()


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    # Allow: python -m src.vision.pipeline [config.yaml]
    cfg_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(cfg_arg)
