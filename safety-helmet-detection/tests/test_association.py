from src.association import associate_people_with_helmets
from src.detector import Detection


def detection(index, label, box, confidence=0.9):
    x1, y1, x2, y2 = box
    return Detection(index, x1, y1, x2, y2, confidence, index, label)


def test_helmet_is_associated_with_overlapping_person():
    left_person = detection(0, "person", (20, 10, 120, 210))
    right_person = detection(1, "person", (140, 10, 240, 210))
    helmet = detection(2, "helmet", (155, 15, 220, 65), 0.95)

    left_status, right_status = associate_people_with_helmets(
        [left_person, right_person],
        [helmet],
    )

    assert left_status.status == "UNKNOWN"
    assert right_status.status == "HELMET"
    assert right_status.matched_objects == [helmet]


def test_conflicting_qualified_predictions_are_unknown():
    person = detection(0, "person", (20, 10, 120, 210))
    helmet = detection(1, "helmet", (25, 12, 115, 80), confidence=0.95)
    no_helmet = detection(2, "no_helmet", (30, 12, 110, 70), confidence=0.76)

    status, = associate_people_with_helmets([person], [helmet, no_helmet])

    assert status.status == "UNKNOWN"
    assert status.matched_objects == [helmet, no_helmet]

    no_helmet.confidence = 0.94
    status, = associate_people_with_helmets([person], [helmet, no_helmet])
    assert status.status == "UNKNOWN"
    assert status.matched_objects == [helmet, no_helmet]

    unmatched_status, = associate_people_with_helmets([person], [])
    assert unmatched_status.status == "UNKNOWN"


def test_helmet_prediction_is_assigned_to_only_the_closest_overlapping_person():
    left_person = detection(0, "person", (20, 10, 200, 210))
    right_person = detection(1, "person", (100, 10, 280, 210))
    helmet = detection(2, "Without Helmet", (130, 40, 180, 100), confidence=0.9)

    left_status, right_status = associate_people_with_helmets(
        [left_person, right_person],
        [helmet],
    )

    assert left_status.status == "UNKNOWN"
    assert right_status.status == "NO HELMET"
