"""Central configuration for the TGS application.

All tunables live here as dataclasses with sensible defaults.
CalibrationData supports JSON persistence for cross-session reuse.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass
class ArucoConfig:
    # cv2.aruco.DICT_4X4_50 = 0
    dictionary_id: int = 0
    marker_size_mm: float = 50.0
    # Order: top-left, top-right, bottom-right, bottom-left
    corner_marker_ids: Tuple[int, int, int, int] = (0, 1, 2, 3)


@dataclass
class YoloConfig:
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    target_class_ids: Dict[int, str] = field(default_factory=lambda: {
        39: "bottle",
        41: "cup",
        45: "bowl",
    })
    input_size: int = 640


@dataclass
class StateConfig:
    placement_threshold_px: float = 50.0
    debounce_frames: int = 15  # ~0.5s at 30 FPS
    lost_object_grace_frames: int = 30  # ~1s before "lost"


@dataclass
class VizConfig:
    color_correct: Tuple[int, int, int] = (0, 255, 0)       # BGR green
    color_incorrect: Tuple[int, int, int] = (0, 0, 255)     # BGR red
    color_pending: Tuple[int, int, int] = (0, 255, 255)     # BGR yellow
    color_grid: Tuple[int, int, int] = (255, 200, 100)      # BGR light blue
    color_lost_warning: Tuple[int, int, int] = (0, 128, 255)  # BGR orange
    arrow_thickness: int = 3
    zone_radius: int = 60  # table-space pixels
    zone_thickness: int = 3
    font_scale: float = 0.8
    font_thickness: int = 2
    hud_banner_height: int = 60
    grid_lines: int = 5
    show_debug_info: bool = True


@dataclass
class PerformanceConfig:
    threading_enabled: bool = True
    mediapipe_enabled: bool = False
    max_detection_age_ms: float = 200.0


@dataclass
class CalibrationData:
    """Persisted calibration state for cross-session reuse."""

    table_width: float = 600.0   # table-space width in virtual pixels
    table_height: float = 400.0  # table-space height in virtual pixels
    homography_matrix: Optional[List[List[float]]] = None
    corner_screen_points: Optional[List[List[float]]] = None
    calibrated: bool = False

    def save(self, path: str = "calibration.json") -> None:
        data = {
            "table_width": self.table_width,
            "table_height": self.table_height,
            "homography_matrix": self.homography_matrix,
            "corner_screen_points": self.corner_screen_points,
            "calibrated": self.calibrated,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str = "calibration.json") -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.table_width = data["table_width"]
            self.table_height = data["table_height"]
            self.homography_matrix = data.get("homography_matrix")
            self.corner_screen_points = data.get("corner_screen_points")
            self.calibrated = data.get("calibrated", False)
            return self.calibrated
        except (json.JSONDecodeError, KeyError):
            return False


@dataclass
class TGSConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    aruco: ArucoConfig = field(default_factory=ArucoConfig)
    yolo: YoloConfig = field(default_factory=YoloConfig)
    state: StateConfig = field(default_factory=StateConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    calibration: CalibrationData = field(default_factory=CalibrationData)
