from __future__ import annotations

import pickle
from types import SimpleNamespace

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


def test_demo_scene_generates_statuses_and_frame():
    frame, statuses = CameraDashboard._create_demo_frame(0, 960, 540)

    assert frame.shape == (540, 960, 3)
    assert "NO HELMET" in statuses
    assert "HELMET" in statuses


def test_corrupt_helmet_weights_keep_camera_preview_available(monkeypatch, tmp_path):
    model_path = tmp_path / "bad-model.pt"
    model_path.write_bytes(b"invalid model")

    class BrokenDetector:
        def __init__(self, model_path, confidence_threshold):
            self.model_path = model_path
            self.confidence_threshold = confidence_threshold

        def load(self):
            raise pickle.UnpicklingError("invalid test checkpoint")

    monkeypatch.setattr("src.dashboard.DetectionEngine", BrokenDetector)
    service = CameraDashboard(DetectionConfig(model_path=model_path))

    assert service.detector is None
    assert "invalid test checkpoint" in service.model_message


def test_two_class_model_labels_directly_classify_worker_boxes():
    detector = DetectionEngine.__new__(DetectionEngine)
    with_helmet = Detection(0, 10, 10, 60, 80, 0.91, 0, "With Helmet")
    without_helmet = Detection(1, 80, 10, 140, 80, 0.88, 1, "Without Helmet")

    subjects, statuses_by_detection, statuses = classify_frame_detections(
        detector,
        [with_helmet, without_helmet],
    )

    assert subjects == [with_helmet, without_helmet]
    assert statuses_by_detection == {0: "HELMET", 1: "NO HELMET"}
    assert statuses == ["HELMET", "NO HELMET"]
