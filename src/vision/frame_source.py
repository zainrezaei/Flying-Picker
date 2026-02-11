"""
frame_source.py — Unified frame source for files (and future cameras).

Wraps cv2.VideoCapture and exposes a simple read/release interface.
A future BaslerCameraSource can implement the same interface.
"""

import cv2 as cv


class FrameSource:
    """Reads frames from a video file (or image sequence).

    A future PiCameraSource (using picamera2 + Raspberry Pi Global
    Shutter Camera) can implement the same read()/release() interface
    and be swapped in without changing the detection pipeline.

    Usage:
        src = FrameSource("path/to/video.mov")
        while True:
            frame = src.read()
            if frame is None:
                break
            ...
        src.release()
    """

    def __init__(self, path: str, loop: bool = True):
        """
        Parameters
        ----------
        path : str
            Path to a video file or image.
        loop : bool
            If True, restart the video from the beginning when it ends.
        """
        self._path = path
        self._loop = loop
        self._cap = cv.VideoCapture(path)

        if not self._cap.isOpened():
            raise FileNotFoundError(
                f"Cannot open video source: {path}"
            )

    # ---- properties ------------------------------------------------

    @property
    def fps(self) -> float:
        """Frames per second reported by the source."""
        return self._cap.get(cv.CAP_PROP_FPS) or 30.0

    @property
    def width(self) -> int:
        return int(self._cap.get(cv.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    @property
    def frame_count(self) -> int:
        return int(self._cap.get(cv.CAP_PROP_FRAME_COUNT))

    # ---- core API --------------------------------------------------

    def read(self):
        """Return the next BGR frame, or None if the source is exhausted."""
        ret, frame = self._cap.read()

        if not ret:
            if self._loop:
                self._cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
            if not ret:
                return None

        return frame

    def release(self):
        """Release the underlying capture."""
        self._cap.release()

    # ---- context manager -------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()

    def __repr__(self):
        return (
            f"FrameSource(path={self._path!r}, "
            f"{self.width}x{self.height} @ {self.fps:.1f} fps)"
        )
