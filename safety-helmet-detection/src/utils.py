from __future__ import annotations

from pathlib import Path


def ensure_parent_directory(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    return path_obj


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def box_area(x1: float, y1: float, x2: float, y2: float) -> float:
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def point_in_box(x: float, y: float, x1: float, y1: float, x2: float, y2: float) -> bool:
    return x1 <= x <= x2 and y1 <= y <= y2


def iou(box_a, box_b) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = box_area(*box_a)
    area_b = box_area(*box_b)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union
