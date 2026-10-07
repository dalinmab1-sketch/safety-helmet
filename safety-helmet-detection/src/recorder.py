from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class VideoRecorder:
    def __init__(self, output_dir: str | Path, fps: float):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Recording FPS must be a positive finite number.")
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.writer: cv2.VideoWriter | None = None
        self.output_path: Path | None = None
        self.frame_size: tuple[int, int] | None = None

    def write(self, frame: np.ndarray) -> None:
        if frame.size == 0:
            raise ValueError("Cannot record an empty video frame.")

        height, width = frame.shape[:2]
        if self.writer is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.output_path = self.output_dir / f"camera_{timestamp}.mp4"
            output_path = self.output_path
            self.frame_size = (width, height)
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                self.frame_size,
            )
            if not writer.isOpened():
                writer.release()
                self.output_path = None
                self.frame_size = None
                raise OSError(f"Could not create video recording: {output_path}")
            self.writer = writer

        if (width, height) != self.frame_size:
            raise ValueError(
                f"Video frame size changed from {self.frame_size} to {(width, height)}."
            )
        self.writer.write(frame)

    def release(self) -> Path | None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        return self.output_path
