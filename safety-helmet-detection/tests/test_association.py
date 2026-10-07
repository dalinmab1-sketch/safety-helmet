from src.association import associate_people_with_helmets
from src.detector import Detection


def detection(index, label, box, confidence=0.9):
    x1, y1, x2, y2 = box
    return Detection(index, x1, y1, x2, y2, confidence, index, label)


def test_helmet_is_associated_with_overlapping_person():
    left_person = detection(0, "person", (20, 10, 120, 210))
    right_person = detection(1, "person", (140, 10, 240, 210))
    helmet = detection(2, "helmet", (155, 15, 220, 65), 0.85)

    left_status, right_status = associate_people_with_helmets(
        [left_person, right_person],
        [helmet],
    )

    assert left_status.status == "UNKNOWN"
    assert right_status.status == "HELMET"
    assert right_status.matched_objects == [helmet]


def test_no_helmet_label_takes_precedence_and_no_match_is_unknown():
    person = detection(0, "person", (20, 10, 120, 210))
    helmet = detection(1, "helmet", (25, 12, 115, 80))
    no_helmet = detection(2, "no_helmet", (30, 12, 110, 70))

    status, = associate_people_with_helmets([person], [helmet, no_helmet])

    assert status.status == "NO HELMET"
    unmatched_status, = associate_people_with_helmets([person], [])
    assert unmatched_status.status == "UNKNOWN"
