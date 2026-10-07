from __future__ import annotations

import time

import cv2
import numpy as np

try:
    import winsound
except ImportError:  # pragma: no cover
    winsound = None

from src.alert_manager import AlertManager
from src.camera import CameraSource
from src.config import DetectionConfig
from src.dashboard import run_dashboard
from src.event_logger import EventLogger
from src.recorder import VideoRecorder
from src.statistics import StatisticsCollector


def draw_recording_indicator(frame: np.ndarray) -> None:
    width = frame.shape[1]
    label = "REC"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(label, font, 0.55, 2)[0]
    right = width - 12
    top = 8
    left = right - text_size[0] - 28
    bottom = top + 28
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 190), -1)
    cv2.circle(frame, (left + 11, top + 14), 4, (255, 255, 255), -1)
    cv2.putText(frame, label, (left + 20, top + 20), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def create_demo_frame(frame_index: int, width: int = 960, height: int = 540) -> tuple[np.ndarray, list[str]]:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (35, 40, 50)

    for x in range(0, width, 140):
        cv2.line(frame, (x, 0), (x, height), (60, 60, 70), 1)

    person_positions = [
        (160, 140, 290, 420),
        (420, 110, 540, 430),
        (620, 170, 760, 420),
    ]
    statuses: list[str] = []

    for idx, (x1, y1, x2, y2) in enumerate(person_positions):
        status = "HELMET" if (frame_index + idx) % 5 != 0 else "NO HELMET"
        if idx == 1 and frame_index % 12 == 0:
            status = "UNKNOWN"
        statuses.append(status)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 2)
        cv2.rectangle(frame, (x1 + 12, y1 - 30), (x1 + 80, y1 + 10), (220, 220, 220), -1)
        cv2.circle(frame, ((x1 + x2) // 2, y1 + 40), 28, (200, 200, 200), -1)

        if status == "HELMET":
            cv2.rectangle(frame, (x1 + 8, y1 - 15), (x1 + 62, y1 + 10), (0, 180, 0), -1)
            cv2.putText(frame, "HELMET", (x1 + 6, y1 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        elif status == "NO HELMET":
            cv2.rectangle(frame, (x1 + 8, y1 - 15), (x1 + 62, y1 + 10), (0, 0, 200), -1)
            cv2.putText(frame, "NO HELMET", (x1 + 6, y1 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(frame, (x1 + 8, y1 - 15), (x1 + 62, y1 + 10), (0, 0, 255), 2)
            cv2.line(frame, (x1 + 62, y1 - 15), (x1 + 8, y1 + 10), (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x1 + 8, y1 - 15), (x1 + 62, y1 + 10), (0, 150, 255), -1)
            cv2.putText(frame, "UNKNOWN", (x1 + 6, y1 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, "DEMO MODE - synthetic helmet detection", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return frame, statuses


def run_demo_mode(config: DetectionConfig) -> int:
    alert_manager = AlertManager(
        alert_after_frames=config.alert_after_frames,
        cooldown_seconds=config.alert_cooldown_seconds,
        enabled=config.alert_enabled,
    )
    stats = StatisticsCollector()
    event_logger = EventLogger(config.log_path)
    frame_index = 0

    print("Demo mode enabled: generating synthetic scene without photos or camera.")

    while True:
        frame, statuses = create_demo_frame(frame_index, config.demo_frame_width, config.demo_frame_height)
        frame_index += 1
        stats.update(statuses)
        alert_triggered = alert_manager.update(statuses)

        if alert_triggered:
            print("ALERT: A person without a helmet was detected.")
            if winsound is not None:
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            event_logger.log_violation("NO HELMET", 0.90, source="demo")

        banner_status = "CHECK CAMERA / MODEL"
        banner_color = (0, 165, 255)
        if "NO HELMET" in statuses:
            banner_status = "ALERT: NO HELMET DETECTED"
            banner_color = (0, 0, 220)
        elif "HELMET" in statuses:
            banner_status = "HELMET DETECTED"
            banner_color = (0, 170, 0)

        height, width = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (width, 55), banner_color, -1)
        cv2.putText(frame, banner_status, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        stats_text = (
            f"People: {stats.snapshot().total_people} | Helmet: {stats.snapshot().helmet_count} | "
            f"No Helmet: {stats.snapshot().no_helmet_count} | Unknown: {stats.snapshot().unknown_count}"
        )
        cv2.putText(frame, stats_text, (15, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(config.display_window_name, frame)

        key = cv2.waitKey(700) & 0xFF
        if key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()
    return 0


def run_camera_preview(camera: CameraSource, config: DetectionConfig) -> int:
    print("Camera is live. Model weights are unavailable; showing camera preview only.")
    print("Add trained helmet-detection weights to enable real helmet classification.")
    recorder = VideoRecorder(config.recording_dir, camera.fps or config.recording_fps)
    started_at = time.monotonic()
    frame_count = 0
    fps = 0.0
    exit_code = 0

    try:
        while True:
            frame = camera.read()
            if frame is None:
                print("Error: Camera stopped returning frames. Check the connection and try again.")
                exit_code = 1
                break

            frame_count += 1
            elapsed = time.monotonic() - started_at
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                started_at = time.monotonic()

            height, width = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (width, 54), (0, 140, 220), -1)
            cv2.putText(
                frame,
                "HELMET CHECK OFFLINE - VALID MODEL REQUIRED",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"{width}x{height}  |  {fps:.1f} FPS  |  Press Q or Esc to quit",
                (15, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            draw_recording_indicator(frame)
            try:
                recorder.write(frame)
            except (OSError, ValueError) as exc:
                print(f"Error recording camera video: {exc}")
                exit_code = 1
                break
            cv2.imshow(config.display_window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        recording_path = recorder.release()
        camera.release()
        cv2.destroyAllWindows()

    if recording_path is not None:
        print(f"Camera recording saved to: {recording_path}")
    return exit_code


def main() -> int:
    config = DetectionConfig()
    return run_dashboard(config)


if __name__ == "__main__":
    raise SystemExit(main())
