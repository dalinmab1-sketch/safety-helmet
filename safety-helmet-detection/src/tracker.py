from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class TrackedObject:
    object_id: int
    center: Tuple[float, float]
    last_seen: int = 0


class SimpleTracker:
    def __init__(self, max_age: int = 5):
        self.max_age = max_age
        self._objects: Dict[int, TrackedObject] = {}
        self._next_id = 1

    def update(self, centers: List[Tuple[float, float]]) -> Dict[int, TrackedObject]:
        current = {}
        for center in centers:
            nearest_id = None
            nearest_distance = None
            for object_id, tracked in self._objects.items():
                distance = ((center[0] - tracked.center[0]) ** 2 + (center[1] - tracked.center[1]) ** 2) ** 0.5
                if nearest_distance is None or distance < nearest_distance:
                    nearest_distance = distance
                    nearest_id = object_id
            if nearest_id is not None and (nearest_distance is None or nearest_distance < 80):
                tracked = self._objects[nearest_id]
                tracked.center = center
                tracked.last_seen = 0
                current[nearest_id] = tracked
            else:
                tracked = TrackedObject(object_id=self._next_id, center=center, last_seen=0)
                self._next_id += 1
                current[tracked.object_id] = tracked
        for object_id in list(self._objects):
            if object_id not in current:
                tracked = self._objects[object_id]
                tracked.last_seen += 1
                if tracked.last_seen <= self.max_age:
                    current[object_id] = tracked
        self._objects = current
        return self._objects
