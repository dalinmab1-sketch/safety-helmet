from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .detector import Detection
from .helmet_classifier import classify_status
from .utils import box_area, iou, point_in_box


@dataclass
class PersonStatus:
    person: Detection
    status: str
    matched_objects: list[Detection]
    confidence: float = 0.0


def associate_people_with_helmets(
    people: Sequence[Detection],
    detections: Sequence[Detection],
    positive_threshold: float = 0.90,
    negative_threshold: float = 0.35,
    overlap_threshold: float = 0.10,
) -> list[PersonStatus]:
    matched_by_person: dict[int, list[Detection]] = {person.index: [] for person in people}
    for det in detections:
        if det.label.lower() == "person" or classify_status(det.label) == "UNKNOWN":
            continue
        candidates: list[tuple[float, float, Detection]] = []
        for person in people:
            if iou(person.box, det.box) < overlap_threshold and not point_in_box(*det.center, *person.box):
                continue

            detection_area = box_area(*det.box)
            overlap_ratio = (
                box_area(
                    max(person.x1, det.x1),
                    max(person.y1, det.y1),
                    min(person.x2, det.x2),
                    min(person.y2, det.y2),
                ) / detection_area
                if detection_area > 0
                else 0.0
            )
            distance = (
                (person.center[0] - det.center[0]) ** 2
                + (person.center[1] - det.center[1]) ** 2
            )
            candidates.append((overlap_ratio, -distance, person))
        if candidates:
            _, _, best_person = max(candidates, key=lambda candidate: candidate[:2])
            matched_by_person[best_person.index].append(det)

    statuses: list[PersonStatus] = []
    for person in people:
        matched = matched_by_person[person.index]
        if not matched:
            statuses.append(
                PersonStatus(
                    person=person,
                    status="UNKNOWN",
                    matched_objects=[],
                    confidence=person.confidence,
                )
            )
            continue

        confident_matches = [
            detection
            for detection in matched
            if detection.confidence
            >= (
                positive_threshold
                if classify_status(detection.label) == "HELMET"
                else negative_threshold
            )
        ]
        labels = {classify_status(detection.label) for detection in confident_matches}
        if len(labels) != 1:
            statuses.append(
                PersonStatus(
                    person=person,
                    status="UNKNOWN",
                    matched_objects=confident_matches,
                    confidence=max(
                        (detection.confidence for detection in confident_matches),
                        default=0.0,
                    ),
                )
            )
            continue

        best_detection = max(confident_matches, key=lambda detection: detection.confidence)
        statuses.append(
            PersonStatus(
                person=person,
                status=classify_status(best_detection.label),
                matched_objects=[best_detection],
                confidence=best_detection.confidence,
            )
        )
    return statuses
