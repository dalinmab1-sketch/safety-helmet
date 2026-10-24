from __future__ import annotations

import numpy as np
import pytest

from src.detector import Detection, DetectionEngine
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


def test_detector_adds_person_predictions_from_separate_model(monkeypatch, tmp_path):
    helmet_path = tmp_path / "helmet.pt"
    person_path = tmp_path / "person.pt"
    helmet_path.write_bytes(b"helmet weights")
    person_path.write_bytes(b"person weights")

    class PersonYolo(FakeYolo):
        names = {2: "dog", 7: "person"}

        def __call__(self, frame, **kwargs):
            self.last_kwargs = kwargs
            return [
                FakeResult(
                    [
                        FakeBox(7, 0.81, (12, 18, 95, 205)),
                        FakeBox(7, 0.28, (1, 1, 10, 12)),
                    ]
                )
            ]

    class HelmetYolo(FakeYolo):
        names = {0: "With Helmet", 1: "Without Helmet"}

    def load_model(path):
        return HelmetYolo(path) if str(path) == str(helmet_path) else PersonYolo(path)

    monkeypatch.setattr("src.detector.YOLO", load_model)
    engine = DetectionEngine(helmet_path, person_model_path=person_path)

    assert engine.load()
    detections = engine.detect(np.zeros((240, 320, 3), dtype=np.uint8))
    people = engine.person_detections(detections)

    assert [person.label for person in people] == ["person"]
    assert people[0].confidence == 0.81
    assert engine.person_model is not None
    assert engine.person_model.last_kwargs == {
        "conf": 0.12,
        "classes": [7],
        "iou": 0.4,
        "verbose": False,
    }
    assert engine.model.last_kwargs is None


def test_detector_does_not_fabricate_person_when_person_model_misses():
    class EmptyPersonModel:
        names = {0: "person"}

        def __init__(self, _path):
            self.last_kwargs = None

        def __call__(self, _frame, **kwargs):
            self.last_kwargs = kwargs
            return [FakeResult([])]

    class HelmetModel(FakeYolo):
        names = {0: "With Helmet", 1: "Without Helmet"}

        def __call__(self, _frame, **kwargs):
            self.last_kwargs = kwargs
            return [FakeResult([FakeBox(1, 0.68, (40, 20, 180, 190))])]

    engine = DetectionEngine("helmet.pt", person_model_path="person.pt")
    engine.model = HelmetModel("helmet.pt")
    engine.person_model = EmptyPersonModel("person.pt")
    engine.names = HelmetModel.names

    detections = engine.detect(np.zeros((240, 320, 3), dtype=np.uint8))

    assert detections == []


def test_helmet_model_is_not_used_as_a_person_detector():
    class HelmetModel(FakeYolo):
        names = {0: "With Helmet", 1: "Without Helmet"}

        def __call__(self, _frame, **kwargs):
            self.last_kwargs = kwargs
            return [
                FakeResult(
                    [
                        FakeBox(0, 0.88, (30, 20, 150, 180)),
                        FakeBox(1, 0.72, (34, 22, 148, 178)),
                    ]
                )
            ]

    engine = DetectionEngine("unused.pt")
    engine.model = HelmetModel("unused.pt")
    engine.names = HelmetModel.names

    detections = engine.detect(np.zeros((240, 320, 3), dtype=np.uint8))

    assert detections == []


def test_status_detection_uses_full_frame_and_filters_by_class_confidence():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    class HeadModel:
        names = {0: "With Helmet", 1: "Without Helmet", 2: "person"}

        def __call__(self, image, **kwargs):
            assert image is frame
            assert kwargs == {"conf": 0.35, "verbose": False}
            return [
                FakeResult(
                    [
                        FakeBox(0, 0.82, (5, 4, 70, 35)),
                        FakeBox(1, 0.82, (8, 6, 65, 34)),
                        FakeBox(2, 0.99, (0, 0, 200, 200)),
                    ]
                )
            ]

    engine = DetectionEngine(
        "unused.pt",
        helmet_positive_threshold=0.8,
        helmet_negative_threshold=0.35,
    )
    engine.model = HeadModel()
    engine.names = HeadModel.names

    detections = engine.detect_status_detections(frame)

    assert [item.label for item in detections] == ["With Helmet", "Without Helmet"]
    assert detections[0].box == (5.0, 4.0, 70.0, 35.0)
    assert detections[1].box == (8.0, 6.0, 65.0, 34.0)


def test_status_detection_rejects_weak_positive_evidence():
    class HeadModel:
        names = {0: "With Helmet"}

        def __call__(self, _frame, **kwargs):
            return [FakeResult([FakeBox(0, 0.89, (5, 4, 70, 35))])]

    engine = DetectionEngine("unused.pt")
    engine.model = HeadModel()
    engine.names = HeadModel.names

    detections = engine.detect_status_detections(np.zeros((200, 200, 3), dtype=np.uint8))

    assert detections == []


def test_detector_rejects_model_without_supported_status_classes(monkeypatch, tmp_path):
    model_path = tmp_path / "person-only.pt"
    model_path.write_bytes(b"fake weights")

    class PersonOnlyModel:
        names = {0: "person"}

        def __init__(self, _path):
            pass

    monkeypatch.setattr("src.detector.YOLO", PersonOnlyModel)
    engine = DetectionEngine(model_path, person_model_path=None)

    with pytest.raises(ValueError, match="no supported helmet-status classes"):
        engine.load()


def test_helmet_label_normalization_and_classification():
    assert normalize_label("NO-HELMET") == "no helmet"
    assert classify_status("no_helmet") == "NO HELMET"
    assert classify_status("safety_helmet") == "HELMET"
    assert classify_status("With Helmet") == "HELMET"
    assert classify_status("Without Helmet") == "NO HELMET"
    assert classify_status("Hardhat") == "HELMET"
    assert classify_status("NO-Hardhat") == "NO HELMET"
    assert classify_status("person") == "UNKNOWN"
