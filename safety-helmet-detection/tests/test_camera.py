from __future__ import annotations

import numpy as np

from src import camera as camera_module
from src.camera import CameraSource


class FakeCapture:
    def __init__(self, source: int, working_source: int):
        self.source = source
        self.working_source = working_source
        self.released = False

    def isOpened(self) -> bool:
        return self.source == self.working_source

    def set(self, _property: int, _value: int) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.source != self.working_source:
            return False, None
        return True, np.zeros((8, 8, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True

    def get(self, _property: int) -> float:
        return 30.0


def test_camera_tries_other_device_indexes_when_default_is_unavailable(monkeypatch):
    captures: list[FakeCapture] = []

    def create_capture(source: int, _backend: int) -> FakeCapture:
        capture = FakeCapture(source, working_source=1)
        captures.append(capture)
        return capture

    monkeypatch.setattr(camera_module.cv2, "VideoCapture", create_capture)
    camera = CameraSource(source=0)

    assert camera.open()
    assert camera.selected_source == 1
    assert camera.read() is not None
    assert camera.is_open()

    camera.release()
    assert not camera.is_open()
    assert camera.selected_source is None


def test_camera_releases_failed_capture_attempts(monkeypatch):
    captures: list[FakeCapture] = []

    def create_capture(source: int, _backend: int) -> FakeCapture:
        capture = FakeCapture(source, working_source=-1)
        captures.append(capture)
        return capture

    monkeypatch.setattr(camera_module.cv2, "VideoCapture", create_capture)
    camera = CameraSource(source=0)

    assert not camera.open()
    assert camera.selected_source is None
    assert all(capture.released for capture in captures)


def test_camera_flips_first_and_subsequent_frames_horizontally(monkeypatch):
    frames = [
        np.array([[[10, 0, 0], [20, 0, 0]]], dtype=np.uint8),
        np.array([[[30, 0, 0], [40, 0, 0]]], dtype=np.uint8),
    ]

    class StreamingCapture(FakeCapture):
        def read(self) -> tuple[bool, np.ndarray | None]:
            return True, frames.pop(0)

    capture = StreamingCapture(source=0, working_source=0)
    monkeypatch.setattr(camera_module.cv2, "VideoCapture", lambda *_args: capture)
    camera = CameraSource(source=0)

    assert camera.open()
    first_frame = camera.read()
    second_frame = camera.read()

    assert first_frame is not None
    assert second_frame is not None
    assert first_frame[0, :, 0].tolist() == [20, 10]
    assert second_frame[0, :, 0].tolist() == [40, 30]


def test_camera_requests_target_frame_rate(monkeypatch):
    requested_properties: dict[int, int | float] = {}

    class ConfigurableCapture(FakeCapture):
        def set(self, property_id: int, value: int | float) -> bool:
            requested_properties[property_id] = value
            return True

    capture = ConfigurableCapture(source=0, working_source=0)
    monkeypatch.setattr(camera_module.cv2, "VideoCapture", lambda *_args: capture)
    camera = CameraSource(source=0, target_fps=60.0)

    assert camera.open()
    assert requested_properties[camera_module.cv2.CAP_PROP_FPS] == 60.0
    camera.release()


def test_video_file_is_not_mirrored_and_stream_urls_are_not_treated_as_files(monkeypatch, tmp_path):
    frame = np.array([[[10, 0, 0], [20, 0, 0]]], dtype=np.uint8)

    class VideoCapture(FakeCapture):
        def read(self) -> tuple[bool, np.ndarray]:
            return True, frame.copy()

    capture = VideoCapture(
        source=str(tmp_path / "clip.mp4"),
        working_source=str(tmp_path / "clip.mp4"),
    )
    monkeypatch.setattr(camera_module.cv2, "VideoCapture", lambda *_args: capture)
    camera = CameraSource(source=str(tmp_path / "clip.mp4"))

    assert camera.is_file_source
    assert camera.open()
    video_frame = camera.read()
    assert video_frame is not None
    assert video_frame[0, :, 0].tolist() == [10, 20]

    camera.release()
    assert not CameraSource(source="rtsp://camera/live").is_file_source
    assert not CameraSource(source="0").is_file_source
