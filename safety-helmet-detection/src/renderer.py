from __future__ import annotations

import cv2
import numpy as np

from .detector import Detection
from .statistics import LiveStatistics


STATUS_COLORS = {
    "HELMET": (0, 170, 0),
    "NO HELMET": (0, 0, 220),
    "UNKNOWN": (0, 215, 255),
}


def _draw_box(frame: np.ndarray, detection: Detection, color: tuple[int, int, int], label: str, score: float | None = None) -> None:
    x1, y1, x2, y2 = map(int, (detection.x1, detection.y1, detection.x2, detection.y2))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = label
    if score is not None:
        text += f" {score:.2f}"
    cv2.putText(frame, text, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def render_frame(
    frame: np.ndarray,
    people: list[Detection],
    status_by_person: dict[int, str],
    all_detections: list[Detection],
    stats: LiveStatistics,
    alert_banner: str = "SYSTEM READY",
    banner_color: tuple[int, int, int] = (0, 165, 255),
) -> np.ndarray:
    annotated = frame.copy()
    person_indices = {person.index for person in people}

    for detection in all_detections:
        if detection.label.lower() == "person" and detection.index in person_indices:
            status = status_by_person.get(detection.index, "UNKNOWN")
            color = STATUS_COLORS.get(status, STATUS_COLORS["UNKNOWN"])
            track_label = f" #{detection.track_id}" if detection.track_id is not None else ""
            _draw_box(
                annotated,
                detection,
                color,
                f"person{track_label}: {status}",
                detection.status_confidence if status != "UNKNOWN" else None,
            )

    height, width = annotated.shape[:2]
    cv2.rectangle(annotated, (0, 0), (width, 58), banner_color, -1)
    font_scale = 0.7
    text_width = cv2.getTextSize(alert_banner, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0][0]
    if text_width > width - 24:
        font_scale *= (width - 24) / text_width
    cv2.putText(
        annotated,
        alert_banner,
        (12, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    violation_rate = (
        f"{stats.violation_rate:.1f}%"
        if stats.violation_rate is not None
        else "N/A"
    )
    stats_text = (
        f"People: {stats.total_people} | Helmet: {stats.helmet_count} | "
        f"No Helmet: {stats.no_helmet_count} | Unknown: {stats.unknown_count} | "
        f"Violation Rate: {violation_rate}"
    )
    cv2.putText(
        annotated,
        stats_text,
        (15, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return annotated
