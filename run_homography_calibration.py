#!/usr/bin/env python3
"""
run_homography_calibration.py — Calibrate pixel → world coordinate transform.

Two modes:

  1. INTERACTIVE (click on image):
     Define known world points, then click the corresponding pixels:
       python run_homography_calibration.py --interactive --source public/IMG_6256.MOV

  2. FROM KNOWN POINTS (no GUI):
     Provide pixel and world points directly:
       python run_homography_calibration.py \\
           --pixel-points "100,50 500,50 500,400 100,400" \\
           --world-points "0,0 400,0 400,300 0,300"

Output: config/homography.json
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.vision.coordinate_transform import (
    calibrate_homography,
    calibrate_homography_interactive,
    save_homography,
    pixel_to_world,
)
from src.vision.frame_source import FrameSource


def _parse_points(s: str) -> np.ndarray:
    """Parse a string like '100,50 500,50 500,400' into an Nx2 array."""
    pairs = s.strip().split()
    points = []
    for pair in pairs:
        x, y = pair.split(",")
        points.append([float(x), float(y)])
    return np.array(points, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate pixel → world (mm) coordinate transform via homography."
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Interactive mode: click reference points on a camera/video frame.",
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Video/camera source for interactive mode.",
    )
    parser.add_argument(
        "--pixel-points", type=str, default=None,
        help='Pixel points as "x1,y1 x2,y2 x3,y3 x4,y4"',
    )
    parser.add_argument(
        "--world-points", type=str, default=None,
        help='World points (mm) as "x1,y1 x2,y2 x3,y3 x4,y4"',
    )
    parser.add_argument(
        "--output", type=str, default="config/homography.json",
        help="Output file path (default: config/homography.json)",
    )

    args = parser.parse_args()

    if args.interactive:
        # --- Interactive mode ---
        if not args.source:
            print("ERROR: --interactive requires --source (video or camera path).")
            sys.exit(1)
        if not args.world_points:
            print("ERROR: --interactive requires --world-points "
                  "(the real-world mm positions of the points you'll click).")
            print('Example: --world-points "0,0 400,0 400,300 0,300"')
            sys.exit(1)

        world_pts = _parse_points(args.world_points)
        source = FrameSource(args.source, loop=False)
        result = calibrate_homography_interactive(source, world_pts)
        source.release()

    else:
        # --- Direct mode ---
        if not args.pixel_points or not args.world_points:
            print("ERROR: Non-interactive mode requires both --pixel-points and --world-points.")
            print('Example:')
            print('  python run_homography_calibration.py \\')
            print('      --pixel-points "100,50 500,50 500,400 100,400" \\')
            print('      --world-points "0,0 400,0 400,300 0,300"')
            sys.exit(1)

        pixel_pts = _parse_points(args.pixel_points)
        world_pts = _parse_points(args.world_points)
        result = calibrate_homography(pixel_pts, world_pts)

    # Save
    save_homography(result, args.output)
    print(f"\n✓ Homography calibration complete! Saved to: {args.output}")
    print(f"  Reprojection error: {result.reprojection_error:.2f} mm")

    # Quick sanity check — transform the calibration points
    print(f"\n  Verification (pixel → world):")
    for px, wd in zip(result.pixel_points, result.world_points):
        wc = pixel_to_world(px[0], px[1], 0.0, result)
        print(f"    pixel ({px[0]:.0f}, {px[1]:.0f}) → "
              f"world ({wc.x_mm:.1f}, {wc.y_mm:.1f}) mm  "
              f"[expected ({wd[0]:.1f}, {wd[1]:.1f})]")


if __name__ == "__main__":
    main()
