from __future__ import annotations

import logging
import math
import pickle
import threading
import time
import webbrowser
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_from_directory, stream_with_context

try:
    import winsound
except ImportError:  # pragma: no cover
    winsound = None

from .alert_manager import AlertManager
from .association import associate_people_with_helmets
from .camera import CameraSource
from .config import DetectionConfig, PROJECT_ROOT
from .detector import Detection, DetectionEngine
from .event_logger import EventLogger
from .recorder import VideoRecorder
from .renderer import render_frame
from .statistics import StatisticsCollector
from .tracker import PersonStatusTracker


class CameraDashboard:
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.camera = CameraSource(
            config.input_source,
            config.camera_width,
            config.camera_height,
            target_fps=config.camera_fps,
        )
        self.detector: DetectionEngine | None = None
        self.model_message = "Helmet model is not installed. Camera preview only."
        model_path = Path(config.model_path)
        if model_path.is_file() and model_path.stat().st_size > 0:
            detector = DetectionEngine(
                model_path,
                config.confidence_threshold,
                config.person_model_path,
                config.person_confidence_threshold,
                config.helmet_positive_threshold,
                config.helmet_negative_threshold,
                config.person_min_area_ratio,
            )
            try:
                if detector.load():
                    self.detector = detector
                    if detector.has_person_detector:
                        self.model_message = "Helmet and person detection models loaded."
                    elif detector.person_model_error:
                        self.model_message = (
                            "Helmet model loaded; person detector could not load: "
                            f"{detector.person_model_error}"
                        )
                    else:
                        self.model_message = (
                            "Helmet model loaded; person detector weights are unavailable."
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
                self.model_message = f"Could not load helmet model: {exc}"

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_jpeg = self._placeholder_jpeg()
        self._record_requested = False
        self._recorder: VideoRecorder | None = None
        self._state: dict[str, object] = {
            "camera_connected": False,
            "demo_mode": config.demo_mode,
            "model_ready": self.detector is not None,
            "model_message": self.model_message,
            "recording": False,
            "recording_requested": False,
            "gesture_control_enabled": config.gesture_control_enabled,
            "gesture": "Waiting for hand",
            "gesture_confidence": 0.0,
            "gesture_action": None,
            "gesture_error": None,
            "recording_path": None,
            "status": "Starting camera",
            "people": 0,
            "helmet": 0,
            "no_helmet": 0,
            "unknown": 0,
            "alerts": 0,
            "fps": 0.0,
            "inference_fps": 0.0,
            "resolution": "--",
            "error": None,
            "updated_at": None,
        }

    @staticmethod
    def _placeholder_jpeg(
        title: str = "Connecting to camera...",
        subtitle: str | None = None,
    ) -> bytes:
        image = np.zeros((480, 800, 3), dtype=np.uint8)
        image[:] = (29, 35, 42)
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        title_x = max((image.shape[1] - title_size[0]) // 2, 12)
        cv2.putText(image, title, (title_x, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 240, 245), 2, cv2.LINE_AA)
        if subtitle:
            subtitle_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
            subtitle_x = max((image.shape[1] - subtitle_size[0]) // 2, 12)
            cv2.putText(image, subtitle, (subtitle_x, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (175, 185, 198), 1, cv2.LINE_AA)
        success, encoded = cv2.imencode(".jpg", image)
        if not success:
            raise RuntimeError("Could not create the dashboard placeholder image.")
        return encoded.tobytes()

    def _show_camera_offline(self, error: str) -> None:
        with self._lock:
            self._state.update(
                {
                    "camera_connected": False,
                    "error": error,
                    "status": "Camera offline - retrying",
                    "fps": 0.0,
                    "inference_fps": 0.0,
                    "resolution": "--",
                }
            )
            self._latest_jpeg = self._placeholder_jpeg(
                "CAMERA OFFLINE",
                "Retrying connection...",
            )

    def _show_video_status(self, status: str, title: str, error: str | None = None) -> None:
        with self._lock:
            self._state.update(
                {
                    "camera_connected": False,
                    "error": error,
                    "status": status,
                    "fps": 0.0,
                    "inference_fps": 0.0,
                    "resolution": "--",
                }
            )
            self._latest_jpeg = self._placeholder_jpeg(title, error)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._camera_loop, name="helmet-camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._recorder is not None:
            saved_path = self._recorder.release()
            with self._lock:
                self._state["recording"] = False
                if saved_path is not None:
                    self._state["recording_path"] = str(saved_path)
            self._recorder = None
        self.camera.release()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return dict(self._state)

    def set_recording(self, enabled: bool) -> None:
        with self._lock:
            self._record_requested = enabled
            self._state["recording_requested"] = enabled

    def set_gesture_control(self, enabled: bool) -> None:
        with self._lock:
            self._state["gesture_control_enabled"] = enabled
            self._state["gesture_action"] = None

    def latest_jpeg(self) -> bytes:
        with self._lock:
            return self._latest_jpeg

    def recordings(self) -> list[dict[str, object]]:
        directory = Path(self.config.recording_dir)
        if not directory.exists():
            return []
        with self._lock:
            active_path = self._recorder.output_path if self._recorder is not None else None
        files = sorted(
            (
                path for path in directory.glob("*.mp4")
                if path.is_file() and path != active_path
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "modified": path.stat().st_mtime,
            }
            for path in files[:20]
        ]

    def _camera_loop(self) -> None:
        if self.config.demo_mode:
            with self._lock:
                self._state["status"] = "Demo mode"

        event_logger = EventLogger(self.config.log_path)
        alert_manager = AlertManager(
            alert_after_frames=self.config.alert_after_frames,
            cooldown_seconds=self.config.alert_cooldown_seconds,
            enabled=self.config.alert_enabled,
        )
        stats = StatisticsCollector()
        tracker = PersonStatusTracker(
            history_frames=self.config.status_history_frames,
            minimum_votes=self.config.status_minimum_votes,
            max_missed_frames=self.config.track_max_missed_frames,
        )
        previous_frame_time = time.monotonic()
        demo_frame_index = 0
        try:
            while not self._stop_event.is_set():
                if not self.config.demo_mode and not self.camera.is_open():
                    if not self.camera.open():
                        if self.camera.is_file_source:
                            self._show_video_status(
                                "VIDEO ERROR",
                                "VIDEO SOURCE ERROR",
                                f"Could not open video file: {self.config.input_source}",
                            )
                            break
                        self._show_camera_offline(
                            f"Cannot open camera or video source: {self.config.input_source}"
                        )
                        self._stop_event.wait(2.0)
                        continue

                    with self._lock:
                        self._state["camera_connected"] = True
                        self._state["error"] = None
                        self._state["status"] = "Camera connected"

                status_by_person: dict[int, str] = {}
                detections = []
                people = []
                statuses: list[str] = []
                inference_seconds = 0.0
                if self.config.demo_mode:
                    frame, statuses = self._create_demo_frame(
                        demo_frame_index,
                        self.config.demo_frame_width,
                        self.config.demo_frame_height,
                    )
                    demo_frame_index += 1
                else:
                    frame = self.camera.read()
                    if frame is None:
                        self.camera.release()
                        if self.camera.is_file_source:
                            self._show_video_status(
                                "VIDEO COMPLETE",
                                "VIDEO COMPLETE",
                            )
                            break
                        self._show_camera_offline("Camera stopped returning frames.")
                        self._stop_event.wait(1.0)
                        continue
                if self.detector is not None and not self.config.demo_mode:
                    inference_started = time.monotonic()
                    detections = self.detector.detect(frame)
                    people, status_by_person, statuses = classify_frame_detections(
                        self.detector,
                        detections,
                        frame,
                    )
                    tracker.update(people, status_by_person)
                    statuses = list(status_by_person.values())
                    inference_seconds = time.monotonic() - inference_started
                    if self.config.diagnostic_mode:
                        logging.getLogger(__name__).info(
                            "Detection diagnostic: helmet_model=%s person_model=%s "
                            "classes=%s person_detections=%s people=%s head_crops=%s "
                            "helmet_predictions=%s statuses=%s",
                            self.detector.model_path,
                            self.detector.person_model_path,
                            self.detector.names,
                            [
                                (item.label, round(item.confidence, 3), item.box)
                                for item in detections
                            ],
                            [
                                (item.track_id, item.box, status_by_person.get(item.index))
                                for item in people
                            ],
                            [
                                (item.label, round(item.confidence, 3), item.box)
                                for item in getattr(self.detector, "last_status_detections", [])
                            ],
                            status_by_person,
                        )
                if self.config.diagnostic_mode and self.detector is not None:
                    logging.getLogger(__name__).info(
                        "Detection inference duration %.3f seconds (%.1f inference FPS)",
                        inference_seconds,
                        1.0 / max(inference_seconds, 1e-6),
                    )
                stats.update(statuses)

                alert_triggered = bool(
                    self.detector is not None
                    and not self.config.demo_mode
                    and alert_manager.update(
                        statuses,
                        [person.track_id for person in people],
                    )
                )
                if alert_triggered:
                    confidence = max(
                        (
                            person.status_confidence
                            for person in people
                            if status_by_person.get(person.index) == "NO HELMET"
                        ),
                        default=0.0,
                    )
                    event_logger.log_violation(
                        "NO HELMET",
                        confidence,
                        source=str(self.config.input_source),
                    )
                    if winsound is not None:
                        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)

                person_detector_ready = bool(
                    self.detector is not None and self.detector.has_person_detector
                )
                banner_status, banner_color = self._banner(
                    people,
                    statuses,
                    model_ready=self.detector is not None,
                    person_detector_ready=person_detector_ready,
                )
                if self.config.demo_mode:
                    banner_status = "DEMO MODE - SYNTHETIC CAMERA SCENE"
                    banner_color = (0, 140, 220)
                annotated = render_frame(
                    frame=frame,
                    people=people,
                    status_by_person=status_by_person,
                    all_detections=detections,
                    stats=stats.snapshot(),
                    alert_banner=banner_status,
                    banner_color=banner_color,
                )
                success, encoded = cv2.imencode(
                    ".jpg",
                    annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, 85],
                )
                if not success:
                    raise RuntimeError("Could not encode a camera frame for the dashboard.")
                resolution = f"{frame.shape[1]} x {frame.shape[0]}"
                with self._lock:
                    self._write_recording(annotated)
                    frame_time = time.monotonic()
                    frame_fps = 1.0 / max(frame_time - previous_frame_time, 1e-6)
                    previous_frame_time = frame_time
                    self._latest_jpeg = encoded.tobytes()
                    live_stats = stats.snapshot()
                    self._state.update(
                        {
                            "status": banner_status,
                            "people": live_stats.total_people,
                            "helmet": live_stats.helmet_count,
                            "no_helmet": live_stats.no_helmet_count,
                            "unknown": live_stats.unknown_count,
                            "fps": round(frame_fps, 1),
                            "inference_fps": (
                                round(1.0 / max(inference_seconds, 1e-6), 1)
                                if inference_seconds > 0 and self.detector is not None
                                else 0.0
                            ),
                            "resolution": resolution,
                            "updated_at": time.time(),
                            "model_ready": self.detector is not None,
                            "model_message": self.model_message,
                            "alerts": int(self._state["alerts"]) + int(alert_triggered),
                        }
                    )
                if self.config.demo_mode:
                    time.sleep(1.0 / self.config.recording_fps)
        except (cv2.error, OSError, RuntimeError, ValueError) as exc:
            with self._lock:
                self._state["error"] = str(exc)
                self._state["status"] = "Camera processing error"
                self._record_requested = False
                self._state["recording_requested"] = False
        finally:
            if self._recorder is not None:
                saved_path = self._recorder.release()
                with self._lock:
                    self._state["recording"] = False
                    if saved_path is not None:
                        self._state["recording_path"] = str(saved_path)
                self._recorder = None
            if not self.config.demo_mode:
                self.camera.release()
            with self._lock:
                self._state["camera_connected"] = False

    @staticmethod
    def _banner(
        people: list,
        statuses: list[str],
        model_ready: bool = True,
        person_detector_ready: bool = True,
    ) -> tuple[str, tuple[int, int, int]]:
        if not model_ready:
            return "DETECTION MODEL UNAVAILABLE - PREVIEW ONLY", (0, 0, 220)
        if not person_detector_ready:
            return "PERSON DETECTOR UNAVAILABLE - STATUS NOT VERIFIED", (0, 0, 220)
        if not people:
            return "NO PERSON DETECTED - STAND IN CAMERA VIEW", (72, 82, 96)
        if len(statuses) != len(people):
            return "HELMET NOT VERIFIED - CHECK HELMET FIT", (0, 215, 255)
        if all(status == "HELMET" for status in statuses):
            return "HELMET DETECTED - YOU APPEAR TO BE WEARING ONE", (0, 170, 0)
        if "NO HELMET" in statuses:
            return "NO HELMET DETECTED - PLEASE PUT ONE ON", (0, 0, 220)
        return "HELMET NOT VERIFIED - CHECK HELMET FIT", (0, 215, 255)

    @staticmethod
    def _create_demo_frame(frame_index: int, width: int, height: int) -> tuple[np.ndarray, list[str]]:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (35, 40, 50)
        for x in range(0, width, 140):
            cv2.line(frame, (x, 0), (x, height), (60, 60, 70), 1)

        positions = [(160, 140, 290, 420), (420, 110, 540, 430), (620, 170, 760, 420)]
        statuses: list[str] = []
        for index, (x1, y1, x2, y2) in enumerate(positions):
            status = "HELMET" if (frame_index + index) % 5 != 0 else "NO HELMET"
            if index == 1 and frame_index % 12 == 0:
                status = "UNKNOWN"
            statuses.append(status)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 2)
            cv2.circle(frame, ((x1 + x2) // 2, y1 + 40), 28, (200, 200, 200), -1)
            color = (0, 180, 0) if status == "HELMET" else (0, 0, 200) if status == "NO HELMET" else (0, 150, 255)
            cv2.rectangle(frame, (x1 + 8, y1 - 15), (x1 + 62, y1 + 10), color, -1)
            cv2.putText(frame, status, (x1 + 6, y1 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        return frame, statuses

    def _write_recording(self, frame: np.ndarray) -> None:
        if self._record_requested:
            if self._recorder is None:
                camera_fps = self.camera.fps
                fps = camera_fps if math.isfinite(camera_fps) and camera_fps > 0 else self.config.recording_fps
                self._recorder = VideoRecorder(self.config.recording_dir, fps)
            width = frame.shape[1]
            cv2.circle(frame, (width - 82, 22), 5, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (width - 70, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            self._recorder.write(frame)
            self._state["recording"] = True
            self._state["recording_path"] = str(self._recorder.output_path)
        else:
            if self._recorder is not None:
                saved_path = self._recorder.release()
                self._recorder = None
                if saved_path is not None:
                    self._state["recording_path"] = str(saved_path)
            self._state["recording"] = False


def classify_frame_detections(
    detector: DetectionEngine,
    detections: Sequence[Detection],
    frame: np.ndarray | None = None,
) -> tuple[list[Detection], dict[int, str], list[str]]:
    people = detector.person_detections(detections)
    if people:
        safety_detections = (
            detector.detect_status_detections(frame)
            if frame is not None and hasattr(detector, "detect_status_detections")
            else detector.safety_detections(detections)
        )
        person_statuses = associate_people_with_helmets(
            people,
            safety_detections,
            positive_threshold=getattr(detector, "helmet_positive_threshold", 0.90),
            negative_threshold=getattr(detector, "helmet_negative_threshold", 0.35),
        )
        for item in person_statuses:
            item.person.status_confidence = item.confidence if item.status != "UNKNOWN" else 0.0
        status_by_person = {item.person.index: item.status for item in person_statuses}
        return people, status_by_person, list(status_by_person.values())

    return [], {}, []


def create_app(service: CameraDashboard) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(PROJECT_ROOT / "templates"),
        static_folder=str(PROJECT_ROOT / "static"),
    )

    @app.get("/")
    def index():
        return render_template("dashboard.html")

    @app.get("/api/status")
    def api_status():
        return jsonify(service.snapshot())

    @app.get("/gesture-model")
    def gesture_model():
        model_path = PROJECT_ROOT / "models" / "gesture_recognizer.task"
        if not model_path.is_file():
            return jsonify({"error": "The hand gesture model is unavailable."}), 404
        return send_from_directory(model_path.parent, model_path.name, mimetype="application/octet-stream")

    @app.post("/api/recording")
    def api_recording():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or type(payload.get("enabled")) is not bool:
            return jsonify({"error": "Send JSON with a boolean 'enabled' field."}), 400
        service.set_recording(payload["enabled"])
        return jsonify(service.snapshot())

    @app.post("/api/gesture-control")
    def api_gesture_control():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or type(payload.get("enabled")) is not bool:
            return jsonify({"error": "Send JSON with a boolean 'enabled' field."}), 400
        service.set_gesture_control(payload["enabled"])
        return jsonify(service.snapshot())

    @app.get("/api/recordings")
    def api_recordings():
        return jsonify(service.recordings())

    @app.get("/recordings/<path:filename>")
    def download_recording(filename: str):
        return send_from_directory(Path(service.config.recording_dir), filename, as_attachment=True)

    @app.get("/video_feed")
    def video_feed():
        def generate() -> Iterator[bytes]:
            last_frame: bytes | None = None
            while True:
                frame = service.latest_jpeg()
                if frame is not last_frame:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                    last_frame = frame
                time.sleep(1.0 / 60.0)

        return Response(stream_with_context(generate()), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


def run_dashboard(config: DetectionConfig) -> int:
    service = CameraDashboard(config)
    service.start()
    app = create_app(service)
    url = f"http://{config.dashboard_host}:{config.dashboard_port}"
    print(f"Helmet Safety dashboard: {url}")
    print("Press Ctrl+C in this terminal to stop the camera and save the recording.")
    threading.Timer(1.0, _open_dashboard, args=(url,)).start()
    try:
        app.run(
            host=config.dashboard_host,
            port=config.dashboard_port,
            debug=False,
            threaded=True,
            use_reloader=False,
        )
    except OSError as exc:
        print(f"Could not start the dashboard at {url}: {exc}")
        return 1
    finally:
        service.stop()
    return 0


def _open_dashboard(url: str) -> None:
    try:
        chrome = webbrowser.get("chrome")
    except webbrowser.Error:
        webbrowser.open_new_tab(url)
    else:
        if not chrome.open_new_tab(url):
            webbrowser.open_new_tab(url)
