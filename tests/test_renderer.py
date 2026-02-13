"""Tests for the AR overlay renderer."""

import numpy as np
import pytest

from tgs.config import VizConfig
from tgs.viz.renderer import Renderer


def test_lerp_color_endpoints():
    red = (0, 0, 255)
    green = (0, 255, 0)
    assert Renderer._lerp_color(red, green, 0.0) == red
    assert Renderer._lerp_color(red, green, 1.0) == green


def test_lerp_color_midpoint():
    black = (0, 0, 0)
    white = (255, 255, 255)
    mid = Renderer._lerp_color(black, white, 0.5)
    assert mid == (127, 127, 127)


def test_lerp_color_clamps():
    a = (0, 0, 0)
    b = (100, 100, 100)
    assert Renderer._lerp_color(a, b, -1.0) == a
    assert Renderer._lerp_color(a, b, 2.0) == b


def test_project_circle_returns_correct_shape(synthetic_marker_frame):
    from tgs.config import ArucoConfig, CalibrationData
    from tgs.perception.surface_tracker import SurfaceTracker

    config = ArucoConfig()
    cal = CalibrationData(table_width=1000.0, table_height=440.0)
    tracker = SurfaceTracker(config, cal)

    frame, _, _ = synthetic_marker_frame
    tracker.detect(frame)

    renderer = Renderer(VizConfig())
    result = renderer._project_circle_to_screen((500, 220), 60.0, tracker, num_points=16)
    assert result is not None
    assert result.shape == (16, 2)


def test_renderer_creates():
    renderer = Renderer(VizConfig())
    assert renderer is not None
