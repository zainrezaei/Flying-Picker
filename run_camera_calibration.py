#!/usr/bin/env python3
"""
run_camera_calibration.py — Calibrate camera intrinsics (lens distortion).

This script handles two modes:

  1. FROM IMAGES (no live camera needed):
     Place checkerboard photos in a folder and run:
       python run_camera_calibration.py --dir path/to/images/

  2. LIVE CAPTURE (when Pi camera is available):
     Run without --dir and it will open the camera for interactive capture:
       python run_camera_calibration.py

Output: config/camera_calibration.json
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.vision.calibration import (
    calibrate_from_images,
    calibrate_live,
    save_calibration,
)
from src.vision.frame_source import FrameSource


def main():
    parser = argparse.ArgumentParser(
        description="Camera intrinsic calibration using a checkerboard pattern."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory containing checkerboard images (.jpg/.png). "
             "If not provided, live capture mode is used.",
    )
    parser.add_argument(
        "--board-cols", type=int, default=9,
        help="Number of inner corners (columns) in the checkerboard (default: 9)",
    )
    parser.add_argument(
        "--board-rows", type=int, default=6,
        help="Number of inner corners (rows) in the checkerboard (default: 6)",
    )
    parser.add_argument(
        "--square-size", type=float, default=25.0,
        help="Size of one checkerboard square in mm (default: 25.0)",
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Video/camera source for live capture mode (default: uses config)",
    )
    parser.add_argument(
        "--output", type=str, default="config/camera_calibration.json",
        help="Output file path (default: config/camera_calibration.json)",
    )
    parser.add_argument(
        "--num-captures", type=int, default=15,
        help="Number of images to capture in live mode (default: 15)",
    )

    args = parser.parse_args()
    board_size = (args.board_cols, args.board_rows)

    if args.dir:
        # --- Mode 1: From images ---
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        image_paths = []
        for pat in patterns:
            image_paths.extend(glob.glob(os.path.join(args.dir, pat)))
        image_paths.sort()

        if not image_paths:
            print(f"ERROR: No images found in '{args.dir}'")
            sys.exit(1)

        print(f"Found {len(image_paths)} images in '{args.dir}'")
        result = calibrate_from_images(
            image_paths,
            board_size=board_size,
            square_size_mm=args.square_size,
        )
    else:
        # --- Mode 2: Live capture ---
        if args.source:
            source = FrameSource(args.source, loop=False)
        else:
            print("ERROR: Live capture mode requires --source (camera or video path).")
            print("       When the Pi camera is ready, pass the camera device.")
            print("       For now, you can test with: --source public/IMG_6256.MOV")
            sys.exit(1)

        result = calibrate_live(
            source,
            board_size=board_size,
            square_size_mm=args.square_size,
            num_captures=args.num_captures,
            save_images_dir="config/calibration_images",
        )
        source.release()

    # Save
    save_calibration(result, args.output)
    print(f"\n✓ Calibration complete! Saved to: {args.output}")
    print(f"  RMS error: {result.rms_error:.4f} (< 0.5 is good)")


if __name__ == "__main__":
    main()
