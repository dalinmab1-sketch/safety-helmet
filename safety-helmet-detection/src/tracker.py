from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Dict, List, Tuple

from .detector import Detection
from .utils import iou

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


@dataclass
class _PersonTrack:
    track_id: int
    box: tuple[float, float, float, float]
    missed_frames: int = 0
    evidence: deque[tuple[str, float]] = field(default_factory=deque)


class PersonStatusTracker:
    def __init__(
        self,
        history_frames: int = 3,
        minimum_votes: int = 2,
        max_missed_frames: int = 5,
    ):
        if history_frames < 1:
            raise ValueError("Status history must contain at least one frame.")
        if not 1 <= minimum_votes <= history_frames:
            raise ValueError("Minimum status votes must fit within the history window.")
        if max_missed_frames < 0:
            raise ValueError("Maximum missed frames cannot be negative.")
        self.history_frames = history_frames
        self.minimum_votes = minimum_votes
        self.max_missed_frames = max_missed_frames
        self._tracks: dict[int, _PersonTrack] = {}
        self._next_id = 1

    def update(self, people: List[Detection], statuses: dict[int, str]) -> list[Detection]:
        candidates: list[tuple[float, float, int, int]] = []
        for person_index, person in enumerate(people):
            for track_id, track in self._tracks.items():
                overlap = iou(person.box, track.box)
                distance = math.dist(person.center, self._center(track.box))
                person_scale = max(person.y2 - person.y1, track.box[3] - track.box[1], 1.0)
                if overlap >= 0.05 or distance <= person_scale * 0.35:
                    candidates.append((overlap, -distance / person_scale, person_index, track_id))
        candidates.sort(reverse=True)

        matched_people: set[int] = set()
        matched_tracks: set[int] = set()
        assignments: dict[int, _PersonTrack] = {}
        for candidate in candidates:
            person_index, track_id = candidate[2], candidate[3]
            if person_index in matched_people or track_id in matched_tracks:
                continue
            matched_people.add(person_index)
            matched_tracks.add(track_id)
            assignments[person_index] = self._tracks[track_id]

        for person_index, person in enumerate(people):
            track = assignments.get(person_index)
            if track is None:
                track = _PersonTrack(
                    track_id=self._next_id,
                    box=person.box,
                    evidence=deque(maxlen=self.history_frames),
                )
                self._tracks[track.track_id] = track
                self._next_id += 1
            matched_tracks.add(track.track_id)
            track.box = person.box
            track.missed_frames = 0

            status = statuses.get(person.index, "UNKNOWN")
            confidence = person.status_confidence
            if status not in {"HELMET", "NO HELMET"}:
                track.evidence.clear()
                smoothed_status = "UNKNOWN"
                smoothed_confidence = 0.0
            else:
                track.evidence.append((status, confidence))
                counts = {
                    candidate: sum(1 for item, _ in track.evidence if item == candidate)
                    for candidate in ("HELMET", "NO HELMET")
                }
                winner = max(counts, key=counts.get)
                if counts[winner] < self.minimum_votes or counts["HELMET"] == counts["NO HELMET"]:
                    smoothed_status = "UNKNOWN"
                    smoothed_confidence = 0.0
                else:
                    smoothed_status = winner
                    smoothed_confidence = sum(
                        score for item, score in track.evidence if item == winner
                    ) / counts[winner]

            person.track_id = track.track_id
            person.status_confidence = smoothed_confidence
            statuses[person.index] = smoothed_status

        for track_id, track in list(self._tracks.items()):
            if track_id in matched_tracks or track_id in {
                item.track_id for item in assignments.values()
            }:
                continue
            track.missed_frames += 1
            track.evidence.clear()
            if track.missed_frames > self.max_missed_frames:
                del self._tracks[track_id]

        return people

    @staticmethod
    def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
