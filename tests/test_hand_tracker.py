"""Tests for the MediaPipe hand tracker wrapper."""

import numpy as np
from tgs.perception.hand_tracker import HandTracker, HandLandmarks


def test_disabled_by_default():
    tracker = HandTracker(enabled=False)
    assert not tracker.enabled


def test_disabled_returns_empty():
    tracker = HandTracker(enabled=False)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = tracker.detect(frame)
    assert result == []


def test_enabled_initializes():
    tracker = HandTracker(enabled=True)
    assert tracker.enabled


def test_no_hands_in_blank_frame():
    tracker = HandTracker(enabled=True)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = tracker.detect(frame)
    assert result == []


def test_fingertip_extraction():
    # Create synthetic hand landmarks
    landmarks = np.zeros((21, 3), dtype=np.float32)
    screen_landmarks = np.zeros((21, 2), dtype=np.float32)
    screen_landmarks[8] = [640.0, 360.0]  # index fingertip

    hand = HandLandmarks(
        landmarks=landmarks, screen_landmarks=screen_landmarks, handedness="Right"
    )
    tracker = HandTracker(enabled=False)
    pos = tracker.get_fingertip_position(hand)
    assert pos == (640, 360)
