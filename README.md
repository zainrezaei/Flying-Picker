# Flying Picker v1

A vision-guided robotic picking system for moving conveyor belts using Basler cameras.

## Overview
This system uses a Basler camera to detect objects on a moving conveyor belt, calculates their position and orientation, and sends tracking data to a robot for dynamic picking.

## Requirements
- Python 3.8+
- Basler camera with pypylon SDK
- OpenCV for image processing
- NumPy for calculations

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure Basler Pylon SDK is installed on your system.

## Usage

Run the main application:
```bash
python src/main.py
```

## Project Structure
- `src/` - Source code
  - `vision/` - Camera and image processing
  - `robot/` - Robot control and tracking
- `config/` - Configuration and calibration data
- `tests/` - Test scripts

## Configuration
Calibration data is stored in `config/calibration_matrix.json`. Run calibration routine before first use.
