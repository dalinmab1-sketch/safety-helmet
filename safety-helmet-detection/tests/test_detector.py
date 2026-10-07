from __future__ import annotations

import numpy as np
import pytest

from src.detector import DetectionEngine
from src.helmet_classifier import classify_status, normalize_label


class FakeBox:
    def __init__(self, class_id, confidence, coordinates):
        self.cls = [class_id]
        self.conf = [confidence]
        self.xyxy = np.array([coordinates], dtype=float)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeYolo:
    names = {0: "person", 1: "helmet", 2: "no_helmet"}

    def __init__(self, _path):
        self.last_kwargs = None

    def __call__(self, frame, **kwargs):
        self.last_kwargs = kwargs
        return [
            FakeResult(
                [
                    FakeBox(0, 0.92, (10, 20, 90, 200)),
                    FakeBox(1, 0.83, (20, 15, 70, 60)),
                ]
            )
        ]


def test_detector_loads_names_and_converts_predictions(monkeypatch, tmp_path):
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"fake weights")
    monkeypatch.setattr("src.detector.YOLO", FakeYolo)
    engine = DetectionEngine(model_path, confidence_threshold=0.6)

    assert engine.load()
    detections = engine.detect(np.zeros((240, 320, 3), dtype=np.uint8))

    assert [item.label for item in detections] == ["person", "helmet"]
    assert detections[0].box == (10.0, 20.0, 90.0, 200.0)
    assert engine.person_detections(detections) == [detections[0]]
    assert engine.safety_detections(detections) == [detections[1]]


def test_detector_missing_model_and_unloaded_use_are_explicit(tmp_path):
    engine = DetectionEngine(tmp_path / "missing.pt")
    assert not engine.load()
    with pytest.raises(RuntimeError, match="Model not loaded"):
        engine.detect(np.zeros((20, 20, 3), dtype=np.uint8))


def test_helmet_label_normalization_and_classification():
    assert normalize_label("NO-HELMET") == "no helmet"
    assert classify_status("no_helmet") == "NO HELMET"
    assert classify_status("safety_helmet") == "HELMET"
    assert classify_status("With Helmet") == "HELMET"
    assert classify_status("Without Helmet") == "NO HELMET"
    assert classify_status("Hardhat") == "HELMET"
    assert classify_status("NO-Hardhat") == "NO HELMET"
    assert classify_status("person") == "UNKNOWN"
