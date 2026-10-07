from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from .utils import ensure_parent_directory


class EventLogger:
    def __init__(self, log_path: str | Path):
        self.log_path = ensure_parent_directory(log_path)

    def log_violation(self, status: str, confidence: float, source: str = "camera") -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.log_path.exists()
        with self.log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if not file_exists:
                writer.writerow(["timestamp", "status", "confidence", "source"])
            writer.writerow([datetime.now().isoformat(timespec="seconds"), status, round(confidence, 3), source])
