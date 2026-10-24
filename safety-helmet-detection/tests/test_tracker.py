from src.detector import Detection
from src.tracker import PersonStatusTracker


def person(index=0, box=(10, 10, 110, 210)):
    return Detection(index, *box, 0.9, 0, "person")


def test_tracker_confirms_status_after_consecutive_evidence():
    tracker = PersonStatusTracker(history_frames=3, minimum_votes=2)
    first_person = person()
    first_person.status_confidence = 0.93
    first_statuses = {0: "HELMET"}

    tracker.update([first_person], first_statuses)

    assert first_statuses == {0: "UNKNOWN"}
    assert first_person.track_id == 1

    next_person = person()
    next_person.status_confidence = 0.91
    next_statuses = {0: "HELMET"}
    tracker.update([next_person], next_statuses)

    assert next_statuses == {0: "HELMET"}
    assert next_person.track_id == first_person.track_id


def test_tracker_resets_evidence_on_unknown_and_does_not_reuse_stale_status():
    tracker = PersonStatusTracker(history_frames=3, minimum_votes=2)
    tracker.update([person()], {0: "NO HELMET"})
    tracker.update([person()], {0: "NO HELMET"})

    unknown_statuses = {0: "UNKNOWN"}
    tracker.update([person()], unknown_statuses)
    assert unknown_statuses == {0: "UNKNOWN"}

    after_unknown = {0: "NO HELMET"}
    tracker.update([person()], after_unknown)
    assert after_unknown == {0: "UNKNOWN"}


def test_tracker_assigns_unique_ids_and_expires_missed_tracks():
    tracker = PersonStatusTracker(history_frames=2, minimum_votes=1, max_missed_frames=1)
    people = [person(0, (10, 10, 110, 210)), person(1, (200, 10, 300, 210))]
    statuses = {0: "HELMET", 1: "HELMET"}
    tracker.update(people, statuses)

    first_ids = [item.track_id for item in people]
    assert len(set(first_ids)) == 2

    tracker.update([], {})
    tracker.update([], {})
    later = [person(0, (10, 10, 110, 210))]
    tracker.update(later, {0: "HELMET"})
    assert later[0].track_id not in first_ids
