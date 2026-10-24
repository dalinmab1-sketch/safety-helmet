import pytest

from src.statistics import StatisticsCollector


def test_statistics_count_current_frame_and_excludes_unknown_from_violation_rate():
    stats = StatisticsCollector()

    snapshot = stats.update(["HELMET", "NO HELMET", "UNKNOWN"])

    assert snapshot.total_people == 3
    assert snapshot.helmet_count == 1
    assert snapshot.no_helmet_count == 1
    assert snapshot.unknown_count == 1
    assert snapshot.violation_rate == pytest.approx(50)


def test_empty_frame_resets_live_counts():
    stats = StatisticsCollector()
    stats.update(["HELMET", "NO HELMET"])

    snapshot = stats.update([])

    assert snapshot.total_people == 0
    assert snapshot.helmet_count == 0
    assert snapshot.no_helmet_count == 0
    assert snapshot.unknown_count == 0
    assert snapshot.violation_rate is None
