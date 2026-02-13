"""Tests for the object detector module."""

import time
import numpy as np
import pytest

from tgs.config import YoloConfig, PerformanceConfig
from tgs.perception.object_detector import (
    Detection,
    DetectionResult,
    ObjectDetector,
)


def test_detection_result_get_by_class():
    result = DetectionResult(
        detections=[
            Detection(class_id=41, class_name="cup", confidence=0.9,
                      bbox=(10, 10, 50, 50), center_screen=(30, 30)),
            Detection(class_id=39, class_name="bottle", confidence=0.8,
                      bbox=(100, 100, 150, 150), center_screen=(125, 125)),
            Detection(class_id=41, class_name="cup", confidence=0.7,
                      bbox=(200, 200, 250, 250), center_screen=(225, 225)),
        ]
    )
    cups = result.get_by_class("cup")
    assert len(cups) == 2
    bottles = result.get_by_class("bottle")
    assert len(bottles) == 1
    bowls = result.get_by_class("bowl")
    assert len(bowls) == 0


def test_detection_defaults():
    det = Detection(
        class_id=41, class_name="cup", confidence=0.9,
        bbox=(10, 10, 50, 50), center_screen=(30, 30),
    )
    assert det.center_table is None
    assert det.timestamp == 0.0


def test_empty_detection_result():
    result = DetectionResult()
    assert len(result.detections) == 0
    assert result.inference_time_ms == 0.0


def test_map_to_table(synthetic_marker_frame):
    """Test that map_to_table enriches detections with table coordinates."""
    from tgs.config import ArucoConfig, CalibrationData
    from tgs.perception.surface_tracker import SurfaceTracker

    config = ArucoConfig()
    cal = CalibrationData(table_width=1000.0, table_height=440.0)
    tracker = SurfaceTracker(config, cal)

    frame, _, _ = synthetic_marker_frame
    tracker.detect(frame)

    det = Detection(
        class_id=41, class_name="cup", confidence=0.9,
        bbox=(580, 300, 620, 340), center_screen=(600, 320),
    )
    result = DetectionResult(detections=[det])

    ObjectDetector.map_to_table(result, tracker)
    assert det.center_table is not None
    # Should be somewhere in the interior of the table
    assert 0 < det.center_table[0] < 1000.0
    assert 0 < det.center_table[1] < 440.0


def test_detector_sync_mode_loads_model():
    """Test that the detector can load the model in synchronous mode.

    This test downloads yolov8n.pt on first run (~6MB).
    """
    config = YoloConfig(model_path="yolov8n.pt")
    perf = PerformanceConfig(threading_enabled=False)
    detector = ObjectDetector(config, perf)
    detector.start()

    try:
        # Submit a blank frame - should return empty detections (no objects)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detector.submit_frame(frame)
        result = detector.get_latest_detections()
        assert isinstance(result, DetectionResult)
        assert result.inference_time_ms > 0
    finally:
        detector.stop()


def test_detector_threaded_mode():
    """Test threaded inference returns results."""
    config = YoloConfig(model_path="yolov8n.pt")
    perf = PerformanceConfig(threading_enabled=True)
    detector = ObjectDetector(config, perf)
    detector.start()

    try:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detector.submit_frame(frame)
        # Give the worker time to process
        time.sleep(1.0)
        result = detector.get_latest_detections()
        assert isinstance(result, DetectionResult)
    finally:
        detector.stop()
