#!/usr/bin/env python3
"""
run_vision.py — Entry point for the Flying Picker vision pipeline.

Usage:
    python run_vision.py                     # uses default config
    python run_vision.py config/custom.yaml  # uses a custom config
"""

import sys
import os

# Ensure the project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.vision.pipeline import run_pipeline

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(cfg)
