"""Shared test fixtures for TGS tests."""

import numpy as np
import pytest
import cv2

from tgs.config import TGSConfig


@pytest.fixture
def config():
    return TGSConfig()


@pytest.fixture
def blank_frame():
    """A 720p black frame."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def aruco_dictionary():
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


@pytest.fixture
def synthetic_marker_frame(aruco_dictionary):
    """A 720p frame with 4 ArUco markers at known positions.

    Marker layout (IDs 0-3):
      ID0 (TL)          ID1 (TR)
        +-----------------+
        |                 |
        |     table       |
        |                 |
        +-----------------+
      ID3 (BL)          ID2 (BR)
    """
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 200  # light gray bg
    marker_size = 80  # pixels

    positions = {
        0: (100, 100),    # top-left
        1: (1100, 100),   # top-right
        2: (1100, 540),   # bottom-right
        3: (100, 540),    # bottom-left
    }

    for marker_id, (x, y) in positions.items():
        marker_img = cv2.aruco.generateImageMarker(aruco_dictionary, marker_id, marker_size)
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        frame[y:y + marker_size, x:x + marker_size] = marker_bgr

    return frame, positions, marker_size
