from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np

from src.config import DetectionConfig
from src.dashboard import CameraDashboard, classify_frame_detections, create_app
from src.detector import Detection, DetectionEngine


class FakeDashboardService:
    def __init__(self, recording_dir):
        self.config = SimpleNamespace(recording_dir=recording_dir)
        self.state = {
            "camera_connected": True,
            "model_ready": False,
            "model_message": "Model required",
            "recording": True,
            "recording_requested": True,
            "gesture_control_enabled": True,
            "gesture": "Thumb_Up",
            "gesture_confidence": 0.93,
            "gesture_action": None,
            "gesture_error": None,
            "recording_path": None,
            "status": "Camera preview only",
            "people": 0,
            "helmet": 0,
            "no_helmet": 0,
            "unknown": 0,
            "alerts": 0,
            "fps": 25.0,
            "resolution": "640 x 480",
            "error": None,
        }

    def snapshot(self):
        return dict(self.state)

    def set_recording(self, enabled):
        self.state["recording_requested"] = enabled

    def set_gesture_control(self, enabled):
        self.state["gesture_control_enabled"] = enabled

    def recordings(self):
        return []

    def latest_jpeg(self):
        return b"jpeg-frame"


def test_dashboard_serves_page_status_and_video_feed(tmp_path):
    service = FakeDashboardService(tmp_path)
    client = create_app(service).test_client()

    page = client.get("/")
    assert page.status_code == 200
    assert b"HARDHAT" in page.data
    assert b"Saved recordings" in page.data
    assert b"Touch mode: Off" in page.data
    assert b'id="touch-mode-button"' in page.data
    assert b"Swipe up or down with one finger" in page.data
    assert b'id="gesture-status"' in page.data

    model = client.head("/gesture-model")
    assert model.status_code == 200
    assert model.content_type == "application/octet-stream"
    assert model.content_length > 0

    status = client.get("/api/status")
    assert status.status_code == 200
    assert status.json["camera_connected"] is True
    assert status.json["recording"] is True

    feed = client.get("/video_feed", buffered=False)
    first_chunk = next(feed.response)
    assert b"Content-Type: image/jpeg" in first_chunk
    assert b"jpeg-frame" in first_chunk
    feed.close()


def test_dashboard_recording_control_requires_boolean(tmp_path):
    service = FakeDashboardService(tmp_path)
    client = create_app(service).test_client()

    invalid = client.post("/api/recording", json={"enabled": "false"})
    assert invalid.status_code == 400

    stopped = client.post("/api/recording", json={"enabled": False})
    assert stopped.status_code == 200
    assert stopped.json["recording_requested"] is False


def test_dashboard_gesture_control_requires_boolean(tmp_path):
    service = FakeDashboardService(tmp_path)
    client = create_app(service).test_client()

    invalid = client.post("/api/gesture-control", json={"enabled": 1})
    assert invalid.status_code == 400

    disabled = client.post("/api/gesture-control", json={"enabled": False})
    assert disabled.status_code == 200
    assert disabled.json["gesture_control_enabled"] is False


def test_dashboard_starts_with_recording_paused(tmp_path):
    service = CameraDashboard(DetectionConfig(model_path=tmp_path / "missing-model.pt"))

    assert service.snapshot()["recording"] is False
    assert service.snapshot()["recording_requested"] is False

    service.set_recording(True)

    assert service.snapshot()["recording_requested"] is True


def test_demo_scene_generates_statuses_and_frame():
    frame, statuses = CameraDashboard._create_demo_frame(0, 960, 540)

    assert frame.shape == (540, 960, 3)
    assert "NO HELMET" in statuses
    assert "HELMET" in statuses


def test_corrupt_helmet_weights_keep_camera_preview_available(monkeypatch, tmp_path):
    model_path = tmp_path / "bad-model.pt"
    model_path.write_bytes(b"invalid model")

    class BrokenDetector:
        def __init__(
            self,
            model_path,
            confidence_threshold,
            person_model_path,
            person_confidence_threshold,
            helmet_positive_threshold,
            helmet_negative_threshold,
            person_min_area_ratio,
        ):
            self.model_path = model_path
            self.confidence_threshold = confidence_threshold
            self.person_model_path = person_model_path
            self.person_confidence_threshold = person_confidence_threshold
            self.helmet_positive_threshold = helmet_positive_threshold
            self.helmet_negative_threshold = helmet_negative_threshold
            self.person_min_area_ratio = person_min_area_ratio

        def load(self):
            raise pickle.UnpicklingError("invalid test checkpoint")

    monkeypatch.setattr("src.dashboard.DetectionEngine", BrokenDetector)
    service = CameraDashboard(DetectionConfig(model_path=model_path))

    assert service.detector is None
    assert "invalid test checkpoint" in service.model_message


def test_helmet_detections_never_fabricate_people():
    detector = DetectionEngine.__new__(DetectionEngine)
    with_helmet = Detection(0, 10, 10, 60, 80, 0.91, 0, "With Helmet")
    without_helmet = Detection(1, 80, 10, 140, 80, 0.88, 1, "Without Helmet")

    subjects, statuses_by_detection, statuses = classify_frame_detections(
        detector,
        [with_helmet, without_helmet],
    )

    assert subjects == []
    assert statuses_by_detection == {}
    assert statuses == []


def test_weak_positive_is_ignored_when_confident_negative_evidence_exists():
    detector = DetectionEngine.__new__(DetectionEngine)
    person = Detection(0, 10, 10, 110, 210, 0.92, 0, "person")
    weak_with = Detection(1, 20, 10, 100, 70, 0.42, 1, "With Helmet")
    strong_without = Detection(2, 21, 11, 99, 69, 0.86, 2, "Without Helmet")

    people, statuses_by_person, statuses = classify_frame_detections(
        detector,
        [person, weak_with, strong_without],
    )

    assert people == [person]
    assert statuses_by_person == {0: "NO HELMET"}
    assert statuses == ["NO HELMET"]


def test_banner_never_claims_helmet_when_classification_is_unknown():
    banner, _color = CameraDashboard._banner(
        [Detection(0, 10, 10, 110, 210, 0.92, 0, "person")],
        ["UNKNOWN"],
    )

    assert banner == "HELMET NOT VERIFIED - CHECK HELMET FIT"


def test_banner_distinguishes_model_and_person_detector_failures():
    assert CameraDashboard._banner([], [], model_ready=False)[0].startswith(
        "DETECTION MODEL UNAVAILABLE"
    )
    assert CameraDashboard._banner([], [], person_detector_ready=False)[0].startswith(
        "PERSON DETECTOR UNAVAILABLE"
    )


def test_dashboard_uses_full_frame_status_detections():
    person = Detection(0, 10, 10, 110, 210, 0.92, 0, "person")
    false_full_frame_helmet = Detection(1, 20, 10, 100, 70, 0.96, 1, "With Helmet")
    confirmed_head_result = Detection(2, 20, 10, 100, 70, 0.84, 2, "Without Helmet")

    class CropDetector:
        person_model = object()

        def person_detections(self, _detections):
            return [person]

        def detect_status_detections(self, frame):
            assert frame.shape == (240, 320, 3)
            return [confirmed_head_result]

        def safety_detections(self, _detections):
            return [false_full_frame_helmet]

    people, status_by_person, statuses = classify_frame_detections(
        CropDetector(),
        [person, false_full_frame_helmet],
        frame=np.zeros((240, 320, 3), dtype=np.uint8),
    )

    assert people == [person]
    assert status_by_person == {0: "NO HELMET"}
    assert statuses == ["NO HELMET"]


def test_safety_banner_uses_neutral_yellow_and_violation_priority_colors():
    no_people, neutral = CameraDashboard._banner([], [])
    uncertain, yellow = CameraDashboard._banner(
        [Detection(0, 10, 10, 110, 210, 0.92, 0, "person")],
        ["UNKNOWN"],
    )
    violation, red = CameraDashboard._banner(
        [
            Detection(0, 10, 10, 110, 210, 0.92, 0, "person"),
            Detection(1, 120, 10, 220, 210, 0.92, 0, "person"),
        ],
        ["UNKNOWN", "NO HELMET"],
    )

    assert no_people.startswith("NO PERSON DETECTED")
    assert neutral == (72, 82, 96)
    assert uncertain.startswith("HELMET NOT VERIFIED")
    assert yellow == (0, 215, 255)
    assert violation.startswith("NO HELMET DETECTED")
    assert red == (0, 0, 220)


def test_safety_banner_never_claims_green_without_a_status_for_each_person():
    people = [
        Detection(0, 10, 10, 110, 210, 0.92, 0, "person"),
        Detection(1, 120, 10, 220, 210, 0.92, 0, "person"),
    ]

    banner, color = CameraDashboard._banner(people, ["HELMET"])

    assert banner.startswith("HELMET NOT VERIFIED")
    assert color == (0, 215, 255)


def test_dashboard_marks_video_end_instead_of_restarting_or_reporting_camera_error(
    monkeypatch,
    tmp_path,
):
    class FileCamera:
        is_file_source = True

        def __init__(self):
            self.opened = False
            self.read_count = 0

        def is_open(self):
            return self.opened

        def open(self):
            self.opened = True
            return True

        def read(self):
            self.read_count += 1
            if self.read_count == 1:
                return np.zeros((120, 160, 3), dtype=np.uint8)
            return None

        def release(self):
            self.opened = False

        @property
        def fps(self):
            return 30.0

    monkeypatch.setattr("src.dashboard.CameraSource", lambda *_args, **_kwargs: FileCamera())
    service = CameraDashboard(
        DetectionConfig(
            input_source=str(tmp_path / "clip.mp4"),
            model_path=tmp_path / "missing-model.pt",
        )
    )

    service._camera_loop()
    state = service.snapshot()

    assert state["status"] == "VIDEO COMPLETE"
    assert state["error"] is None
    assert state["camera_connected"] is False
