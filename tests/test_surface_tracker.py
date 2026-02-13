"""Tests for ArUco surface tracking and homography computation."""

import numpy as np
import cv2
import pytest

from tgs.config import ArucoConfig, CalibrationData
from tgs.perception.surface_tracker import SurfaceTracker


@pytest.fixture
def tracker():
    config = ArucoConfig()
    calibration = CalibrationData(table_width=1000.0, table_height=440.0)
    return SurfaceTracker(config, calibration)


def test_detect_all_four_markers(tracker, synthetic_marker_frame):
    frame, positions, marker_size = synthetic_marker_frame
    assert tracker.detect(frame) is True
    assert tracker.is_valid()
    assert tracker.frames_since_lost == 0


def test_homography_not_none_after_detect(tracker, synthetic_marker_frame):
    frame, _, _ = synthetic_marker_frame
    tracker.detect(frame)
    assert tracker.get_homography() is not None
    assert tracker.get_inverse_homography() is not None


def test_screen_corners_returned(tracker, synthetic_marker_frame):
    frame, _, _ = synthetic_marker_frame
    tracker.detect(frame)
    corners = tracker.get_screen_corners()
    assert corners is not None
    assert corners.shape == (4, 2)


def test_screen_to_table_roundtrip(tracker, synthetic_marker_frame):
    """screen->table->screen should be near-identity."""
    frame, _, _ = synthetic_marker_frame
    tracker.detect(frame)

    # Pick a point somewhere in the middle of the table area
    screen_pt = np.array([[600.0, 320.0]])
    table_pt = tracker.screen_to_table(screen_pt)
    assert table_pt is not None

    recovered = tracker.table_to_screen(table_pt)
    assert recovered is not None
    np.testing.assert_allclose(recovered, screen_pt, atol=2.0)


def test_table_corners_map_correctly(tracker, synthetic_marker_frame):
    """Table-space corners (0,0), (W,0), (W,H), (0,H) should map back to screen marker positions."""
    frame, _, _ = synthetic_marker_frame
    tracker.detect(frame)

    tw = 1000.0
    th = 440.0
    table_corners = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32)
    screen_pts = tracker.table_to_screen(table_corners)
    assert screen_pts is not None

    # They should be near the marker positions (within reason for the inner corner)
    detected_corners = tracker.get_screen_corners()
    np.testing.assert_allclose(screen_pts, detected_corners, atol=5.0)


def test_no_markers_returns_false(tracker, blank_frame):
    assert tracker.detect(blank_frame) is False
    assert not tracker.is_valid()


def test_stale_homography_preserved(tracker, synthetic_marker_frame, blank_frame):
    """After losing markers, cached homography is still usable within grace period."""
    frame, _, _ = synthetic_marker_frame
    tracker.detect(frame)
    assert tracker.is_valid()

    # Lose markers for a few frames
    for _ in range(10):
        tracker.detect(blank_frame)

    assert tracker.is_valid()  # still within grace period
    assert tracker.is_stale
    assert tracker.get_homography() is not None


def test_homography_expires_after_max_stale(tracker, synthetic_marker_frame, blank_frame):
    frame, _, _ = synthetic_marker_frame
    tracker.detect(frame)

    # Lose markers for longer than grace period (90 frames)
    for _ in range(100):
        tracker.detect(blank_frame)

    assert not tracker.is_valid()
    assert tracker.get_homography() is None
    assert tracker.screen_to_table(np.array([[100, 100]])) is None


def test_calibration_persistence(synthetic_marker_frame, tmp_path):
    """Detect markers, save calibration, load into new tracker."""
    config = ArucoConfig()
    cal = CalibrationData(table_width=1000.0, table_height=440.0)
    tracker = SurfaceTracker(config, cal)

    frame, _, _ = synthetic_marker_frame
    tracker.detect(frame)

    cal_path = str(tmp_path / "cal.json")
    cal.save(cal_path)

    # New tracker loads saved calibration
    cal2 = CalibrationData()
    cal2.load(cal_path)
    tracker2 = SurfaceTracker(config, cal2)

    # Should be immediately valid (from saved homography)
    assert tracker2.is_valid()
    assert tracker2.get_homography() is not None
