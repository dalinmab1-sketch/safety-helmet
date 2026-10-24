from __future__ import annotations

from collections.abc import Iterator
import math

import cv2
import numpy as np


class CameraSource:
    def __init__(
        self,
        source: int | str = 0,
        width: int | None = None,
        height: int | None = None,
        target_fps: float | None = 60.0,
    ):
        self.source = source
        self.width = width
        self.height = height
        if target_fps is not None and (not math.isfinite(target_fps) or target_fps <= 0):
            raise ValueError("Target camera FPS must be finite and positive.")
        self.target_fps = target_fps
        self.capture: cv2.VideoCapture | None = None
        self._first_frame: np.ndarray | None = None
        self.selected_source: int | str | None = None
        self._mirror_frames = False

    def open(self) -> bool:
        self.release()
        capture_source = (
            int(self.source)
            if isinstance(self.source, str) and self.source.isdigit()
            else self.source
        )

        for source in self._sources(capture_source):
            for backend in self._backends(source):
                capture = cv2.VideoCapture(source, backend)
                if not capture.isOpened():
                    capture.release()
                    continue

                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if self.width is not None:
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                if self.height is not None:
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                if self.target_fps is not None:
                    capture.set(cv2.CAP_PROP_FPS, self.target_fps)

                for _ in range(3):
                    success, frame = capture.read()
                    if success and frame is not None and frame.size > 0:
                        self.capture = capture
                        self.selected_source = source
                        self._mirror_frames = isinstance(source, int)
                        self._first_frame = frame
                        return True

                capture.release()

        self.capture = None
        self.selected_source = None
        self._first_frame = None
        self._mirror_frames = False
        return False

    @property
    def is_file_source(self) -> bool:
        if not isinstance(self.source, str) or self.source.isdigit():
            return False
        return not self.source.lower().startswith(
            ("rtsp://", "rtsps://", "rtmp://", "http://", "https://")
        )

    @staticmethod
    def _sources(source: int | str) -> Iterator[int | str]:
        yield source
        if isinstance(source, int):
            for index in range(5):
                if index != source:
                    yield index

    @staticmethod
    def _backends(source: int | str) -> Iterator[int]:
        if isinstance(source, int):
            if hasattr(cv2, "CAP_DSHOW"):
                yield cv2.CAP_DSHOW
            if hasattr(cv2, "CAP_MSMF"):
                yield cv2.CAP_MSMF
        yield cv2.CAP_ANY

    def read(self) -> np.ndarray | None:
        if self.capture is None:
            return None
        if self._first_frame is not None:
            frame = self._first_frame
            self._first_frame = None
            return self._prepare_frame(frame)
        success, frame = self.capture.read()
        if not success or frame is None or frame.size == 0:
            return None
        return self._prepare_frame(frame)

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        return cv2.flip(frame, 1) if self._mirror_frames else frame

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.selected_source = None
        self._first_frame = None
        self._mirror_frames = False

    def is_open(self) -> bool:
        return self.capture is not None and self.capture.isOpened()

    @property
    def fps(self) -> float:
        if self.capture is None:
            return 0.0
        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        return fps if math.isfinite(fps) and fps > 0 else 0.0
