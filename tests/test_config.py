"""Tests for TGS configuration."""

import json
import os
import tempfile

from tgs.config import (
    CalibrationData,
    CameraConfig,
    ArucoConfig,
    TGSConfig,
    YoloConfig,
)


def test_default_config_creates():
    config = TGSConfig()
    assert config.camera.width == 1280
    assert config.camera.height == 720
    assert config.camera.fps == 30


def test_aruco_defaults():
    config = ArucoConfig()
    assert config.dictionary_id == 0  # DICT_4X4_50
    assert config.corner_marker_ids == (0, 1, 2, 3)


def test_yolo_target_classes():
    config = YoloConfig()
    assert 39 in config.target_class_ids  # bottle
    assert 41 in config.target_class_ids  # cup
    assert 45 in config.target_class_ids  # bowl
    assert len(config.target_class_ids) == 3


def test_calibration_save_load():
    cal = CalibrationData(
        table_width=800.0,
        table_height=500.0,
        homography_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        corner_screen_points=[[0, 0], [800, 0], [800, 500], [0, 500]],
        calibrated=True,
    )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        cal.save(path)
        assert os.path.exists(path)

        loaded = CalibrationData()
        assert loaded.load(path) is True
        assert loaded.table_width == 800.0
        assert loaded.table_height == 500.0
        assert loaded.calibrated is True
        assert loaded.homography_matrix == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    finally:
        os.unlink(path)


def test_calibration_load_missing_file():
    cal = CalibrationData()
    assert cal.load("/nonexistent/path.json") is False


def test_calibration_load_corrupt_json():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        f.write("not valid json{{{")
        path = f.name

    try:
        cal = CalibrationData()
        assert cal.load(path) is False
    finally:
        os.unlink(path)


def test_config_nested_defaults():
    config = TGSConfig()
    assert config.performance.threading_enabled is True
    assert config.performance.mediapipe_enabled is False
    assert config.state.debounce_frames == 15
    assert config.viz.show_debug_info is True
