from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DetectionConfig:
    model_path: str | Path = PROJECT_ROOT / "models" / "models.helmet_model.pt"
    person_model_path: str | Path | None = PROJECT_ROOT / "yolov8n.pt"
    input_source: str | int = 0
    confidence_threshold: float = 0.45
    person_confidence_threshold: float = 0.12
    helmet_positive_threshold: float = 0.90
    helmet_negative_threshold: float = 0.35
    person_min_area_ratio: float = 0.005
    status_history_frames: int = 3
    status_minimum_votes: int = 2
    track_max_missed_frames: int = 5
    diagnostic_mode: bool = False
    alert_after_frames: int = 3
    alert_cooldown_seconds: float = 5.0
    alert_enabled: bool = True
    snapshot_dir: str | Path = PROJECT_ROOT / "snapshots"
    recording_dir: str | Path = PROJECT_ROOT / "recordings"
    recording_fps: float = 20.0
    gesture_control_enabled: bool = True
    gesture_confidence_threshold: float = 0.75
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 5000
    log_path: str | Path = PROJECT_ROOT / "logs" / "violations.csv"
    display_window_name: str = "Safety Helmet Detection - press Q to quit"
    camera_width: int | None = None
    camera_height: int | None = None
    camera_fps: float = 60.0
    show_stats: bool = True
    demo_mode: bool = False
    demo_frame_width: int = 960
    demo_frame_height: int = 540

    def as_runtime(self) -> "DetectionConfig":
        return self
