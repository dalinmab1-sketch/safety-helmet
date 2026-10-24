from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Sequence

import numpy as np
from ultralytics import YOLO

from .helmet_classifier import classify_status


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
    track_id: int | None = None
    status_confidence: float = 0.0

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class DetectionEngine:
    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.45,
        person_model_path: str | Path | None = None,
        person_confidence_threshold: float = 0.12,
        helmet_positive_threshold: float = 0.90,
        helmet_negative_threshold: float = 0.35,
        person_min_area_ratio: float = 0.005,
    ):
        for name, threshold in (
            ("confidence_threshold", confidence_threshold),
            ("person_confidence_threshold", person_confidence_threshold),
            ("helmet_positive_threshold", helmet_positive_threshold),
            ("helmet_negative_threshold", helmet_negative_threshold),
            ("person_min_area_ratio", person_min_area_ratio),
        ):
            if not 0 <= threshold <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.person_model_path = Path(person_model_path) if person_model_path is not None else None
        self.person_confidence_threshold = person_confidence_threshold
        self.helmet_positive_threshold = helmet_positive_threshold
        self.helmet_negative_threshold = helmet_negative_threshold
        self.person_min_area_ratio = person_min_area_ratio
        self.model: YOLO | None = None
        self.person_model: YOLO | None = None
        self.person_model_error: str | None = None
        self.person_class_ids: list[int] = []
        self.names: dict[int, str] = {}
        self.last_status_detections: list[Detection] = []

    def load(self) -> bool:
        if not self.model_path.exists() or self.model_path.stat().st_size == 0:
            return False
        self.model = YOLO(str(self.model_path))
        self.names = getattr(self.model, "names", {}) or {}
        if not any(classify_status(name) != "UNKNOWN" for name in self.names.values()):
            raise ValueError(
                "Helmet model has no supported helmet-status classes. "
                f"Found classes: {self.names}"
            )
        if self.person_model_path is not None and self.person_model_path.is_file():
            try:
                self.person_model = YOLO(str(self.person_model_path))
                person_names = getattr(self.person_model, "names", {}) or {}
                self.person_class_ids = [
                    class_id
                    for class_id, name in person_names.items()
                    if str(name).casefold() == "person"
                ]
                if not self.person_class_ids:
                    raise ValueError(
                        "Person model has no class named 'person'. "
                        f"Found classes: {person_names}"
                    )
            except (
                AttributeError,
                EOFError,
                ImportError,
                IndexError,
                KeyError,
                OSError,
                pickle.UnpicklingError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                self.person_model = None
                self.person_class_ids = []
                self.person_model_error = str(exc)
        return True

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() before detect().")

        detections: list[Detection] = []
        if self.person_model is not None:
            person_results = self.person_model(
                frame,
                conf=self.person_confidence_threshold,
                classes=self.person_class_ids,
                iou=0.4,
                verbose=False,
            )
            person_detections: list[Detection] = []
            self._append_detections(
                person_detections,
                person_results,
                getattr(self.person_model, "names", {}) or {},
            )
            minimum_area = frame.shape[0] * frame.shape[1] * self.person_min_area_ratio
            detections.extend(
                detection
                for detection in person_detections
                if (detection.x2 - detection.x1) * (detection.y2 - detection.y1) >= minimum_area
            )
        else:
            if not self.has_person_detector:
                return []
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            self._append_detections(detections, results, self.names)
        return detections

    @property
    def has_person_detector(self) -> bool:
        return self.person_model is not None or any(
            label.lower() == "person" for label in self.names.values()
        )

    def detect_status_detections(
        self,
        frame: np.ndarray,
    ) -> list[Detection]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() before detect_status_detections().")

        results = self.model(
            frame,
            conf=min(self.helmet_positive_threshold, self.helmet_negative_threshold),
            verbose=False,
        )
        detections: list[Detection] = []
        self._append_detections(detections, results, self.names)
        detections = [
            detection
            for detection in detections
            if detection.confidence
            >= (
                self.helmet_positive_threshold
                if classify_status(detection.label) == "HELMET"
                else self.helmet_negative_threshold
            )
            and classify_status(detection.label) != "UNKNOWN"
        ]
        self.last_status_detections = detections
        return detections

    @staticmethod
    def _append_detections(
        detections: list[Detection],
        results: Sequence[object],
        names: dict[int, str],
    ) -> None:
        if not results:
            return
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return
        for box in boxes:
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            class_id = int(box.cls[0])
            detections.append(
                Detection(
                    index=len(detections),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(box.conf[0]),
                    class_id=class_id,
                    label=str(names.get(class_id, f"class_{class_id}")),
                )
            )

    def person_detections(self, detections: Sequence[Detection]) -> list[Detection]:
        return [d for d in detections if d.label.lower() == "person"]

    def safety_detections(self, detections: Sequence[Detection]) -> list[Detection]:
        return [d for d in detections if d.label.lower() != "person"]
