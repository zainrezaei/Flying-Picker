# Flying Picker — Vision System

A computer-vision pipeline that detects a rectangular object on a conveyor belt, extracts its **position (x, y)** and **orientation (angle)**, and displays the results in a live overlay window. Designed as the "eyes" of a robotic pick-and-place system.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [How It Works — Step by Step](#how-it-works--step-by-step)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [What Has Been Done](#what-has-been-done)
- [What Still Needs to Be Done](#what-still-needs-to-be-done)

---

## Overview

The **Flying Picker** is a robotic pick-and-place system. A camera watches a conveyor belt, detects objects, and sends their position + angle to a robot so it can pick them mid-flight.

### Target Hardware

| # | Item | Purpose | Est. Cost |
|---|------|---------|----------|
| 1 | **Raspberry Pi 5 (8 GB RAM)** | Compute — runs the vision pipeline | €99 |
| 2 | Official USB-C Power Supply (27 W) | Power — Pi 5 requires the official 27 W PD supply | €20.20 |
| 3 | Active Cooler for Pi 5 | Cooling — prevents thermal throttling during vision processing | €10.10 |
| 4 | **Raspberry Pi Global Shutter Camera (Sony IMX296)** | The Eye — global shutter eliminates motion blur / jelly effect on moving conveyor | €85 |
| 5 | 6 mm CS-Mount Lens | The Optic — wide-angle lens to see the full belt width | €30 |
| 6 | Camera Cable (Mini → Standard) | Pi 5 uses a Mini CSI connector; adapter cable needed | €3 |
| 7 | MicroSD Card (32 GB, A2 Class) | Storage — fast boot + program loading | €25.30 |
| 8 | LED Strip (12 V Cool White) | Lighting — consistent illumination for reliable detection | €12.99 |
| | | **Total** | **€285.59** |

Currently the vision system runs on a **pre-recorded video** (`puplic/IMG_6256.MOV`) of a light-coloured square card on a dark surface. No camera hardware is needed yet — the architecture is designed so the Pi Global Shutter Camera can be swapped in later (via `picamera2`) without changing the detection logic.

---

## Project Structure

```
flying-picker-v1/
│
├── run_vision.py                  # ← Entry point — run this
│
├── config/
│   └── vision_config.yaml         # All tunable parameters (thresholds, colors, paths)
│
├── src/
│   ├── vision/
│   │   ├── __init__.py            # Package exports
│   │   ├── frame_source.py        # FrameSource class — reads frames from video/camera
│   │   ├── preprocess.py          # Grayscale → blur → threshold → morphology
│   │   ├── detection.py           # Contour detection → minAreaRect → (x, y, angle)
│   │   ├── pipeline.py            # Main loop — ties everything together + live overlay
│   │   └── Image_ReadWrite.py     # Legacy/reference OpenCV experiments (not used by pipeline)
│   │
│   └── robot/                     # (empty — future robot communication module)
│
├── tests/                         # (empty — future unit/integration tests)
│
├── puplic/
│   ├── IMG_6256.MOV               # Sample video of the object on a surface
│   ├── 01_Flying picker.pdf       # Project overview document
│   ├── Flying Picker research paper.pdf
│   └── Vision_System_BOM.pdf      # Bill of Materials for vision hardware
│
└── requirements.txt               # Python dependencies
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        run_vision.py                             │
│                      (entry point)                               │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                     pipeline.py                                  │
│              (main loop — orchestrator)                          │
│                                                                  │
│   Loads config ──→ Opens FrameSource ──→ Loop:                   │
│                                           │                      │
│        ┌──────────────────────────────────┘                      │
│        │                                                         │
│        ▼                                                         │
│   ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐     │
│   │ FrameSource │──▶│ preprocess() │──▶│ detect_object()  │     │
│   │             │   │              │   │                  │     │
│   │ Read next   │   │ BGR→Gray     │   │ Find contours   │     │
│   │ video frame │   │ Gaussian blur│   │ Largest contour │     │
│   │             │   │ Binary thresh│   │ minAreaRect()   │     │
│   │ (loops at   │   │ Morph close  │   │                  │     │
│   │  end of     │   │              │   │ Returns:         │     │
│   │  video)     │   │ Returns:     │   │  center (x, y)  │     │
│   │             │   │  binary mask │   │  size (w, h)     │     │
│   └─────────────┘   └──────────────┘   │  angle (θ)      │     │
│                                         │  box corners    │     │
│                                         └───────┬─────────┘     │
│                                                 │               │
│                                                 ▼               │
│                                        ┌────────────────┐       │
│                                        │ Draw overlay   │       │
│                                        │ on frame:      │       │
│                                        │ • Green box    │       │
│                                        │ • Red centroid │       │
│                                        │ • Angle text   │       │
│                                        │                │       │
│                                        │ cv2.imshow()   │       │
│                                        │ Print to console│      │
│                                        └────────────────┘       │
└──────────────────────────────────────────────────────────────────┘

                  ▲
                  │  All parameters loaded from:
                  │
        ┌─────────────────────┐
        │ vision_config.yaml  │
        │                     │
        │ • video file path   │
        │ • blur kernel size  │
        │ • threshold value   │
        │ • min contour area  │
        │ • overlay colors    │
        └─────────────────────┘
```

---

## How It Works — Step by Step

### 1. Frame Acquisition (`frame_source.py`)

`FrameSource` wraps OpenCV's `VideoCapture`. It opens the video file, reads one frame at a time, and **loops** back to frame 0 when the video ends (so you get a continuous stream for testing).

```
Video file (.MOV) ──▶ FrameSource.read() ──▶ BGR frame (numpy array)
```

### 2. Preprocessing (`preprocess.py`)

Each raw BGR frame goes through four steps to produce a clean **binary mask**:

| Step | Function | Why |
|------|----------|-----|
| **Grayscale** | `cv2.cvtColor(BGR2GRAY)` | Reduce 3 channels to 1 |
| **Gaussian Blur** | `cv2.GaussianBlur(kernel=5)` | Remove noise / smooth edges |
| **Binary Threshold** | `cv2.threshold(150)` | Light card → white (255), dark background → black (0) |
| **Morphological Close** | `cv2.morphologyEx(MORPH_CLOSE)` | Fill small holes/gaps inside the object |

**Input:** full-colour frame → **Output:** binary mask (white object on black)

### 3. Detection (`detection.py`)

The binary mask is analysed to find the object's pose:

| Step | Function | Output |
|------|----------|--------|
| Find all contours | `cv2.findContours()` | List of contour point arrays |
| Pick the largest | `max(contours, key=cv2.contourArea)` | Single contour |
| Filter by min area | Area check (default 5000 px²) | Rejects noise |
| Fit rotated rectangle | `cv2.minAreaRect()` | Center (x, y), size (w, h), angle (θ) |
| Get box corners | `cv2.boxPoints()` | 4 corner coordinates for drawing |

Returns a `DetectionResult` dataclass:
```
DetectionResult(
    center_x,      # centroid X in pixels
    center_y,      # centroid Y in pixels
    width,          # bounding box width
    height,         # bounding box height
    angle,          # rotation in degrees
    contour,        # raw contour points
    box_points      # 4 corners of the rotated box
)
```

### 4. Overlay & Display (`pipeline.py`)

The pipeline draws on the original frame and shows it:

- **Green rotated bounding box** — the detected rectangle outline
- **Red dot** — centroid (x, y)
- **White text** — `x=... y=... angle=...°`

Two windows are shown:
- **"Flying Picker — Detection"** — original frame with overlay
- **"Flying Picker — Mask"** — the binary preprocessing result

Console output each frame:
```
[frame   42]  x=  330.4  y=  586.7  angle=   2.3°  (0.8 ms)
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- macOS / Linux / Windows

### Install

```bash
cd flying-picker-v1

# Create virtual environment (if not already done)
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
python run_vision.py
```

Two OpenCV windows will open showing the live detection. Press **`q`** to quit.

### Run with a custom config

```bash
python run_vision.py path/to/custom_config.yaml
```

---

## Configuration

All tunable parameters are in `config/vision_config.yaml`:

```yaml
input:
  video_path: "puplic/IMG_6256.MOV"    # Video source

preprocess:
  blur_kernel_size: 5       # Gaussian blur kernel (must be odd)
  threshold_value: 150      # Binary threshold cutoff (0–255)
  threshold_max: 255        # Value for pixels above threshold

detection:
  min_contour_area: 5000    # Ignore contours smaller than this (px²)

display:
  show_windows: true        # Show live OpenCV windows
  show_mask: true           # Also show the binary mask window
  overlay_color: [0, 255, 0]  # Bounding box (BGR green)
  centroid_color: [0, 0, 255] # Centroid dot (BGR red)
  text_color: [255, 255, 255] # Label text (BGR white)
```

**Tuning tips:**
- If the mask has noise (white specks in background) → **increase** `threshold_value`
- If the object is missing parts in the mask → **decrease** `threshold_value`
- If small contours are being detected as the object → **increase** `min_contour_area`

---

## What Has Been Done

### ✅ Vision Pipeline (complete for file-based input)

| Component | File | Status |
|-----------|------|--------|
| Frame source (video file) | `src/vision/frame_source.py` | ✅ Done |
| Preprocessing (threshold) | `src/vision/preprocess.py` | ✅ Done |
| Object detection (contour) | `src/vision/detection.py` | ✅ Done |
| Main pipeline + live overlay | `src/vision/pipeline.py` | ✅ Done |
| Configuration system | `config/vision_config.yaml` | ✅ Done |
| Entry point script | `run_vision.py` | ✅ Done |
| Bug fixes in legacy code | `src/vision/Image_ReadWrite.py` | ✅ Fixed |
| Dependencies | `requirements.txt` | ✅ Updated |

### Current capabilities

- Reads video at native FPS (~30 fps)
- Detects a single light rectangular object on a dark background
- Extracts centroid (x, y) and rotation angle (θ) per frame
- Draws live overlay (bounding box + centroid + angle label)
- Loops video for continuous testing
- All parameters configurable via YAML (no code changes needed)
- ~1 ms processing time per frame

---

## What Still Needs to Be Done

### 🔲 Phase 1 — Camera Integration (Raspberry Pi 5)

| Task | Details |
|------|--------|
| **Pi Global Shutter Camera capture** | Implement a `PiCameraSource` class using `picamera2` that matches the `FrameSource` interface. The Sony IMX296 global shutter sensor eliminates motion blur on the moving conveyor. Swap it into the pipeline via config. |
| **Camera calibration** | Intrinsic calibration (lens distortion correction for the 6 mm CS-mount lens) using a checkerboard pattern + `cv2.calibrateCamera()`. |
| **Pi 5 optimisation** | Ensure the pipeline runs efficiently on Pi 5's ARM CPU. Profile memory usage (8 GB available) and consider GPU acceleration via OpenCV's UMat if needed. |

### 🔲 Phase 2 — Coordinate Mapping

| Task | Details |
|------|---------|
| **Pixel → world transform** | Compute a homography matrix to map pixel (x, y) to real-world conveyor coordinates (mm). Requires a calibration procedure with known reference points. |
| **Conveyor tracking** | Sync detection with conveyor movement — either via encoder input or frame-to-frame object tracking — so the robot knows where the object *will be* at pick time. |

### 🔲 Phase 3 — Robot Communication

| Task | Details |
|------|---------|
| **Serial protocol** | Define the message format for sending `(x, y, θ)` to the robot controller over `pyserial`. |
| **`src/robot/` module** | Implement serial connection, message packaging, and send/receive logic. |
| **Timing / sync** | Ensure the detection → send → pick timing is tight enough for the conveyor speed. |

### 🔲 Phase 4 — Robustness & Testing

| Task | Details |
|------|---------|
| **Multi-object detection** | Extend `detect_object()` to return a list of results instead of just the largest. |
| **Adaptive thresholding** | Handle varying lighting conditions (e.g., `cv2.adaptiveThreshold` or HSV-based segmentation). |
| **Unit tests** | Add tests in `tests/` for preprocessing, detection, and coordinate transforms. |
| **Logging** | Replace `print()` with Python `logging` module for configurable verbosity. |
| **Error handling** | Graceful recovery from dropped frames, serial disconnects, etc. |

### 🔲 Phase 5 — Documentation & Polish

| Task | Details |
|------|---------|
| **Calibration guide** | Step-by-step instructions for camera + homography calibration. |
| **Deployment guide** | How to run on the actual conveyor setup. |
| **Performance profiling** | Benchmark end-to-end latency under real conditions. |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | latest | Array operations |
| `opencv-python` | latest | Computer vision (capture, threshold, contours, display) |
| `pyyaml` | ≥6.0 | Configuration file loading |
| `picamera2` | — | Raspberry Pi camera library *(commented out — enable when running on Pi 5)* |
| `pyserial` | — | Robot serial communication *(commented out — enable when hardware available)* |

---

## License

*(To be determined)*
