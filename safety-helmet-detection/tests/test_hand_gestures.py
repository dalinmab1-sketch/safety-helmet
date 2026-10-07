from __future__ import annotations

import numpy as np

from src.hand_gestures import GestureCommandDebouncer, HandGestureRecognizer


def test_recording_gestures_need_a_hold_and_do_not_repeat():
    control = GestureCommandDebouncer(hold_seconds=1.0, release_seconds=0.5)

    assert control.update("Thumb_Up", now=0.0) is None
    assert control.update("Thumb_Up", now=0.9) is None
    assert control.update("Thumb_Up", now=1.0) is True
    assert control.update("Thumb_Up", now=2.0) is None
    assert control.update(None, now=2.1) is None
    assert control.update(None, now=2.7) is None
    assert control.update("Open_Palm", now=2.8) is None
    assert control.update("Open_Palm", now=3.8) is False


def test_hand_gesture_model_loads_and_processes_frame():
    recognizer = HandGestureRecognizer()
    try:
        gesture, confidence, action = recognizer.process(np.zeros((240, 320, 3), dtype=np.uint8))
    finally:
        recognizer.close()

    assert gesture is None
    assert confidence == 0.0
    assert action is None
