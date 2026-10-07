from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    index: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    label: str

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class DetectionEngine:
    def __init__(self, model_path: str | Path, confidence_threshold: float = 0.45):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.model: YOLO | None = None
        self.names: dict[int, str] = {}

    def load(self) -> bool:
        if not self.model_path.exists() or self.model_path.stat().st_size == 0:
            return False
        self.model = YOLO(str(self.model_path))
        self.names = getattr(self.model, "names", {}) or {}
        return True

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() before detect().")

        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        if not results:
            return []

        result = results[0]
        detections: list[Detection] = []
        if result.boxes is None:
            return detections

        for index, box in enumerate(result.boxes):
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            class_id = int(box.cls[0])
            label = str(self.names.get(class_id, f"class_{class_id}"))
            confidence = float(box.conf[0])
            detections.append(
                Detection(
                    index=index,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=confidence,
                    class_id=class_id,
                    label=label,
                )
            )
        return detections

    def person_detections(self, detections: Sequence[Detection]) -> list[Detection]:
        return [d for d in detections if d.label.lower() == "person"]

    def safety_detections(self, detections: Sequence[Detection]) -> list[Detection]:
        return [d for d in detections if d.label.lower() != "person"]
