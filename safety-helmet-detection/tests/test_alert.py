from src.alert_manager import AlertManager


def test_alert_waits_for_streak_and_respects_cooldown(monkeypatch):
    now = 10.0
    monkeypatch.setattr("src.alert_manager.time.monotonic", lambda: now)
    alerts = AlertManager(alert_after_frames=3, cooldown_seconds=5.0)

    assert not alerts.update(["NO HELMET"])
    assert not alerts.update(["NO HELMET"])
    assert alerts.update(["NO HELMET"])
    assert not alerts.update(["NO HELMET"])
    assert not alerts.update(["HELMET"])
    assert not alerts.update(["NO HELMET"])
    assert not alerts.update(["NO HELMET"])
    assert not alerts.update(["NO HELMET"])

    now += 5.1
    assert alerts.update(["NO HELMET"])


def test_alert_is_disabled_and_empty_status_resets_streak():
    disabled = AlertManager(alert_after_frames=1, enabled=False)
    assert not disabled.update(["NO HELMET"])

    alerts = AlertManager(alert_after_frames=2, cooldown_seconds=0)
    assert not alerts.update(["NO HELMET"])
    assert not alerts.update([])
    assert not alerts.update(["NO HELMET"])
    assert alerts.update(["NO HELMET"])
