from __future__ import annotations

import time
from typing import Sequence


class AlertManager:
    def __init__(self, alert_after_frames: int = 3, cooldown_seconds: float = 5.0, enabled: bool = True):
        self.alert_after_frames = alert_after_frames
        self.cooldown_seconds = cooldown_seconds
        self.enabled = enabled
        self.violation_streak = 0
        self.last_alert_time = 0.0

    def update(self, statuses: Sequence[str]) -> bool:
        if not statuses:
            self.violation_streak = 0
            return False

        has_violation = any(status == "NO HELMET" for status in statuses)
        if has_violation:
            self.violation_streak += 1
        else:
            self.violation_streak = 0

        now = time.monotonic()
        should_alert = (
            self.enabled
            and has_violation
            and self.violation_streak >= self.alert_after_frames
            and now - self.last_alert_time >= self.cooldown_seconds
        )

        if should_alert:
            self.last_alert_time = now
            self.violation_streak = 0
            return True

        return False
