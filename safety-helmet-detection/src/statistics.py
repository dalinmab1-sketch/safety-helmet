from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class LiveStatistics:
    total_people: int = 0
    helmet_count: int = 0
    no_helmet_count: int = 0
    unknown_count: int = 0
    violation_rate: float = 0.0


class StatisticsCollector:
    def __init__(self):
        self.total_people = 0
        self.helmet_count = 0
        self.no_helmet_count = 0
        self.unknown_count = 0
        self.frame_people = 0
        self.violation_rate = 0.0

    def update(self, statuses: Sequence[str]) -> LiveStatistics:
        self.frame_people = len(statuses)
        self.helmet_count = sum(1 for status in statuses if status == "HELMET")
        self.no_helmet_count = sum(1 for status in statuses if status == "NO HELMET")
        self.unknown_count = sum(1 for status in statuses if status == "UNKNOWN")
        self.total_people += self.frame_people
        self.violation_rate = (self.no_helmet_count / self.frame_people) * 100.0 if self.frame_people else 0.0
        return LiveStatistics(
            total_people=self.frame_people,
            helmet_count=self.helmet_count,
            no_helmet_count=self.no_helmet_count,
            unknown_count=self.unknown_count,
            violation_rate=self.violation_rate,
        )

    def snapshot(self) -> LiveStatistics:
        return LiveStatistics(
            total_people=self.frame_people,
            helmet_count=self.helmet_count,
            no_helmet_count=self.no_helmet_count,
            unknown_count=self.unknown_count,
            violation_rate=self.violation_rate,
        )
