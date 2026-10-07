from __future__ import annotations

import numpy as np
import pytest

from src import recorder as recorder_module
from src.recorder import VideoRecorder


class FakeVideoWriter:
    def __init__(self, opened: bool = True):
        self.opened = opened
        self.frames: list[np.ndarray] = []
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.released = True


def test_recorder_writes_video_frames_and_releases_writer(monkeypatch, tmp_path):
    fake_writer = FakeVideoWriter()
    monkeypatch.setattr(recorder_module.cv2, "VideoWriter", lambda *args: fake_writer)
    monkeypatch.setattr(recorder_module.cv2, "VideoWriter_fourcc", lambda *args: 0)
    recorder = VideoRecorder(tmp_path, fps=20.0)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    recorder.write(frame)
    saved_path = recorder.release()

    assert saved_path is not None
    assert saved_path.parent == tmp_path
    assert saved_path.suffix == ".mp4"
    assert len(fake_writer.frames) == 1
    assert fake_writer.released


def test_recorder_reports_video_writer_failure(monkeypatch, tmp_path):
    fake_writer = FakeVideoWriter(opened=False)
    monkeypatch.setattr(recorder_module.cv2, "VideoWriter", lambda *args: fake_writer)
    monkeypatch.setattr(recorder_module.cv2, "VideoWriter_fourcc", lambda *args: 0)
    recorder = VideoRecorder(tmp_path, fps=20.0)

    with pytest.raises(OSError, match="Could not create video recording"):
        recorder.write(np.zeros((240, 320, 3), dtype=np.uint8))

    assert fake_writer.released
    assert recorder.release() is None
