from __future__ import annotations

import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision

from .config import PROJECT_ROOT


GESTURE_ACTIONS = {
    "Thumb_Up": True,
    "Open_Palm": False,
}


class GestureCommandDebouncer:
    def __init__(self, hold_seconds: float = 1.0, release_seconds: float = 0.5):
        if hold_seconds <= 0 or release_seconds <= 0:
            raise ValueError("Gesture hold and release times must be positive.")
        self.hold_seconds = hold_seconds
        self.release_seconds = release_seconds
        self._candidate: str | None = None
        self._candidate_since = 0.0
        self._latched: str | None = None
        self._last_seen = 0.0

    def update(self, gesture: str | None, now: float | None = None) -> bool | None:
        current_time = time.monotonic() if now is None else now
        if gesture not in GESTURE_ACTIONS:
            self._candidate = None
            if current_time - self._last_seen >= self.release_seconds:
                self._latched = None
            return None

        self._last_seen = current_time
        if gesture != self._candidate:
            self._candidate = gesture
            self._candidate_since = current_time
            return None

        if gesture == self._latched or current_time - self._candidate_since < self.hold_seconds:
            return None

        self._latched = gesture
        return GESTURE_ACTIONS[gesture]


class HandGestureRecognizer:
    def __init__(self, model_path: str | Path | None = None, confidence_threshold: float = 0.75):
        resolved_model = Path(model_path) if model_path is not None else PROJECT_ROOT / "models" / "gesture_recognizer.task"
        if not resolved_model.is_file() or resolved_model.stat().st_size == 0:
            raise FileNotFoundError(f"Hand gesture model is missing or empty: {resolved_model}")
        if not 0 < confidence_threshold <= 1:
            raise ValueError("Gesture confidence threshold must be between 0 and 1.")

        options = vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(resolved_model)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=confidence_threshold,
            min_hand_presence_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold,
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)
        self._debouncer = GestureCommandDebouncer()
        self.confidence_threshold = confidence_threshold

    def process(self, frame: np.ndarray) -> tuple[str | None, float, bool | None]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._recognizer.recognize(image)
        if not result.gestures or not result.gestures[0]:
            return None, 0.0, self._debouncer.update(None)

        category = max(result.gestures[0], key=lambda item: item.score)
        gesture = category.category_name
        confidence = float(category.score)
        if gesture not in GESTURE_ACTIONS or confidence < self.confidence_threshold:
            return gesture, confidence, self._debouncer.update(None)
        return gesture, confidence, self._debouncer.update(gesture)

    def close(self) -> None:
        self._recognizer.close()
