from __future__ import annotations

import time
from typing import Sequence


class AlertManager:
    def __init__(self, alert_after_frames: int = 3, cooldown_seconds: float = 5.0, enabled: bool = True):
        if alert_after_frames < 1:
            raise ValueError("Alert frame threshold must be at least one.")
        if cooldown_seconds < 0:
            raise ValueError("Alert cooldown cannot be negative.")
        self.alert_after_frames = alert_after_frames
        self.cooldown_seconds = cooldown_seconds
        self.enabled = enabled
        self.violation_streak = 0
        self.last_alert_time: float | None = None
        self._violation_streaks: dict[object, int] = {}
        self._alerted_violations: set[object] = set()

    def update(
        self,
        statuses: Sequence[str],
        track_ids: Sequence[int | None] | None = None,
    ) -> bool:
        if track_ids is not None and len(track_ids) != len(statuses):
            raise ValueError("Track IDs must correspond to the supplied statuses.")

        active_violations: set[object] = set()
        for index, status in enumerate(statuses):
            if status != "NO HELMET":
                continue
            track_id = track_ids[index] if track_ids is not None else None
            key: object = track_id if track_id is not None else ("untracked", 0)
            active_violations.add(key)

        for key in self._violation_streaks.keys() - active_violations:
            self._alerted_violations.discard(key)
        self._violation_streaks = {
            key: self._violation_streaks.get(key, 0) + 1
            for key in active_violations
        }
        self.violation_streak = max(self._violation_streaks.values(), default=0)

        if not self.enabled:
            return False

        eligible = [
            key
            for key, streak in self._violation_streaks.items()
            if streak >= self.alert_after_frames and key not in self._alerted_violations
        ]
        now = time.monotonic()
        cooldown_elapsed = (
            self.last_alert_time is None
            or now - self.last_alert_time >= self.cooldown_seconds
        )
        if not eligible or not cooldown_elapsed:
            return False

        selected = max(eligible, key=self._violation_streaks.__getitem__)
        self._alerted_violations.add(selected)
        self.last_alert_time = now
        return True
