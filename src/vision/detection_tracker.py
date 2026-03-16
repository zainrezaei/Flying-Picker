"""
detection_tracker.py — Single-send detection with debounce.

Ensures each physical part triggers exactly ONE coordinate send to the
robot, even if the per-frame detection is jittery (detect → lose → detect).

State machine:
    IDLE  →  (N consecutive detections)  →  CONFIRMING
    CONFIRMING  →  (confirmed)  →  SENT        (sends coordinates once)
    SENT  →  (M consecutive no-detections)  →  IDLE
"""

import logging
import math
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional

try:
    from .coordinate_transform import WorldCoordinate
except ImportError:
    from coordinate_transform import WorldCoordinate

logger = logging.getLogger(__name__)


class _State(Enum):
    IDLE = auto()       # No part present, waiting for detections
    CONFIRMING = auto() # Seeing detections, building confidence
    SENT = auto()       # Coordinates sent, suppressing until part leaves


@dataclass
class TrackerResult:
    """Output of DetectionTracker.update()."""
    should_send: bool                       # True exactly once per part
    coord: Optional[WorldCoordinate]        # Averaged coordinates to send


class DetectionTracker:
    """Track part presence and emit send signals exactly once per part.

    Parameters
    ----------
    confirm_frames : int
        Consecutive detection frames required before sending (default 3).
    exit_frames : int
        Consecutive no-detection frames required before resetting (default 5).
    distance_threshold_mm : float
        If a new detection is farther than this from the locked position,
        treat it as a brand-new part (default 20.0 mm).
    """

    def __init__(
        self,
        confirm_frames: int = 3,
        exit_frames: int = 5,
        distance_threshold_mm: float = 20.0,
    ):
        self.confirm_frames = max(1, confirm_frames)
        self.exit_frames = max(1, exit_frames)
        self.distance_threshold_mm = distance_threshold_mm

        self._state = _State.IDLE
        self._detect_count = 0       # consecutive frames with detection
        self._miss_count = 0         # consecutive frames without detection

        # Running sum for averaging during confirmation window
        self._sum_x = 0.0
        self._sum_y = 0.0
        self._sum_angle = 0.0
        self._n_samples = 0

        # Locked position (set once confirmed, used for distance check)
        self._locked_coord: Optional[WorldCoordinate] = None

    # -------------------------------------------------------------- #
    # Public API                                                       #
    # -------------------------------------------------------------- #

    def update(self, coord: Optional[WorldCoordinate]) -> TrackerResult:
        """Process one frame's detection result.

        Parameters
        ----------
        coord : WorldCoordinate or None
            The world-coordinate detection for this frame, or None if
            no object was detected.

        Returns
        -------
        TrackerResult
            .should_send is True exactly once per physical part.
            .coord contains the averaged coordinates to send.
        """

        # ---------- No detection this frame ----------
        if coord is None:
            self._detect_count = 0

            if self._state == _State.CONFIRMING:
                # Lost detection during confirmation → abort
                self._miss_count += 1
                if self._miss_count >= self.exit_frames:
                    logger.info("[TRACKER] Detection lost during confirmation, resetting")
                    self._reset()
                return TrackerResult(should_send=False, coord=None)

            if self._state == _State.SENT:
                self._miss_count += 1
                if self._miss_count >= self.exit_frames:
                    logger.info("[TRACKER] Part exited camera FOV, ready for next part")
                    self._reset()
                return TrackerResult(should_send=False, coord=None)

            # IDLE + no detection → stay idle
            return TrackerResult(should_send=False, coord=None)

        # ---------- Detection this frame ----------
        self._miss_count = 0

        if self._state == _State.IDLE:
            # New detection while idle → start confirming
            self._state = _State.CONFIRMING
            self._detect_count = 1
            self._sum_x = coord.x_mm
            self._sum_y = coord.y_mm
            self._sum_angle = coord.angle_deg
            self._n_samples = 1
            return TrackerResult(should_send=False, coord=None)

        if self._state == _State.CONFIRMING:
            self._detect_count += 1
            self._sum_x += coord.x_mm
            self._sum_y += coord.y_mm
            self._sum_angle += coord.angle_deg
            self._n_samples += 1

            if self._detect_count >= self.confirm_frames:
                # Confirmed! Compute averaged coordinates and send once
                avg_coord = WorldCoordinate(
                    x_mm=self._sum_x / self._n_samples,
                    y_mm=self._sum_y / self._n_samples,
                    angle_deg=self._sum_angle / self._n_samples,
                )
                self._locked_coord = avg_coord
                self._state = _State.SENT
                logger.info(
                    f"[TRACKER] Part confirmed, sending coordinates: "
                    f"x={avg_coord.x_mm:.1f}, y={avg_coord.y_mm:.1f}, "
                    f"angle={avg_coord.angle_deg:.1f}"
                )
                return TrackerResult(should_send=True, coord=avg_coord)

            return TrackerResult(should_send=False, coord=None)

        if self._state == _State.SENT:
            # Check if this detection has jumped far enough to be a new part
            if self._locked_coord is not None:
                dist = math.hypot(
                    coord.x_mm - self._locked_coord.x_mm,
                    coord.y_mm - self._locked_coord.y_mm,
                )
                if dist > self.distance_threshold_mm:
                    logger.info(
                        f"[TRACKER] New part detected (distance={dist:.1f} mm > "
                        f"threshold={self.distance_threshold_mm:.1f} mm), "
                        f"starting new confirmation"
                    )
                    self._reset()
                    # Immediately start confirming the new part
                    self._state = _State.CONFIRMING
                    self._detect_count = 1
                    self._sum_x = coord.x_mm
                    self._sum_y = coord.y_mm
                    self._sum_angle = coord.angle_deg
                    self._n_samples = 1
                    return TrackerResult(should_send=False, coord=None)

            # Same part, still visible → suppress
            return TrackerResult(should_send=False, coord=None)

        return TrackerResult(should_send=False, coord=None)

    # -------------------------------------------------------------- #
    # Internal                                                         #
    # -------------------------------------------------------------- #

    def _reset(self):
        """Reset all state to IDLE."""
        self._state = _State.IDLE
        self._detect_count = 0
        self._miss_count = 0
        self._sum_x = 0.0
        self._sum_y = 0.0
        self._sum_angle = 0.0
        self._n_samples = 0
        self._locked_coord = None

    @property
    def state(self) -> str:
        """Current state name (for debugging)."""
        return self._state.name
