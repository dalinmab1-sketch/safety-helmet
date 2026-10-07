from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .detector import Detection
from .helmet_classifier import classify_status
from .utils import iou, point_in_box


@dataclass
class PersonStatus:
    person: Detection
    status: str
    matched_objects: list[Detection]
    confidence: float = 0.0


def associate_people_with_helmets(people: Sequence[Detection], detections: Sequence[Detection], overlap_threshold: float = 0.10) -> list[PersonStatus]:
    statuses: list[PersonStatus] = []

    for person in people:
        relevant: list[Detection] = []
        for det in detections:
            if det.label.lower() == "person":
                continue
            if iou(person.box, det.box) >= overlap_threshold or point_in_box(*person.center, *det.box):
                relevant.append(det)

        if not relevant:
            statuses.append(PersonStatus(person=person, status="UNKNOWN", matched_objects=[], confidence=person.confidence))
            continue

        labels = [classify_status(det.label) for det in relevant]
        if "NO HELMET" in labels:
            status = "NO HELMET"
        elif "HELMET" in labels:
            status = "HELMET"
        else:
            status = "UNKNOWN"

        statuses.append(PersonStatus(person=person, status=status, matched_objects=relevant, confidence=person.confidence))

    return statuses
