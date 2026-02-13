"""ArUco marker detection and homography-based surface tracking.

Detects 4 ArUco markers at table corners to compute a homography matrix
that maps between screen pixel coordinates and a virtual table-space
coordinate system.
"""

from __future__ import annotations

import numpy as np
import cv2

from tgs.config import ArucoConfig, CalibrationData


# For each corner marker, which corner of the marker polygon to use
# as the reference point (the corner closest to the table interior).
# ArUco corners are ordered: top-left, top-right, bottom-right, bottom-left
# of the marker itself.
_INNER_CORNER_INDEX = {
    0: 2,  # TL marker -> use its bottom-right corner
    1: 3,  # TR marker -> use its bottom-left corner
    2: 0,  # BR marker -> use its top-left corner
    3: 1,  # BL marker -> use its top-right corner
}


class SurfaceTracker:
    """Detects ArUco markers and computes/caches the homography matrix."""

    def __init__(
        self,
        aruco_config: ArucoConfig,
        calibration: CalibrationData,
    ) -> None:
        self._config = aruco_config
        self._calibration = calibration
        self._dictionary = cv2.aruco.getPredefinedDictionary(aruco_config.dictionary_id)
        params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(self._dictionary, params)
        self._corner_ids = aruco_config.corner_marker_ids

        self._homography: np.ndarray | None = None
        self._inv_homography: np.ndarray | None = None
        self._screen_corners: np.ndarray | None = None
        self._frames_since_lost: int = 0
        self._max_stale_frames: int = 90  # 3 seconds at 30 FPS

        # Load saved calibration if available
        if calibration.homography_matrix is not None:
            self._homography = np.array(calibration.homography_matrix, dtype=np.float64)
            self._inv_homography = np.linalg.inv(self._homography)
            if calibration.corner_screen_points is not None:
                self._screen_corners = np.array(
                    calibration.corner_screen_points, dtype=np.float32
                )

    def detect(self, frame: np.ndarray) -> bool:
        """Detect markers in frame. Returns True if all 4 corners found.

        On success: updates homography and resets stale counter.
        On failure: increments stale counter, preserves last valid homography.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)

        if ids is None:
            self._frames_since_lost += 1
            return False

        ids_flat = ids.flatten()
        marker_map: dict[int, np.ndarray] = {}

        for i, marker_id in enumerate(ids_flat):
            mid = int(marker_id)
            if mid in self._corner_ids:
                corner_idx = _INNER_CORNER_INDEX.get(mid, 0)
                marker_map[mid] = corners[i][0][corner_idx]

        if len(marker_map) != 4:
            self._frames_since_lost += 1
            return False

        # All 4 markers found — compute homography
        tw = self._calibration.table_width
        th = self._calibration.table_height

        src = np.array(
            [marker_map[mid] for mid in self._corner_ids], dtype=np.float32
        )
        dst = np.array(
            [[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32
        )

        H, mask = cv2.findHomography(src, dst)
        if H is None:
            self._frames_since_lost += 1
            return False

        self._homography = H
        self._inv_homography = np.linalg.inv(H)
        self._screen_corners = src.copy()
        self._frames_since_lost = 0

        # Persist calibration
        self._calibration.homography_matrix = H.tolist()
        self._calibration.corner_screen_points = src.tolist()
        self._calibration.calibrated = True

        return True

    def get_homography(self) -> np.ndarray | None:
        """Returns screen->table homography (3x3), or None if stale/uncalibrated."""
        if self._homography is None:
            return None
        if self._frames_since_lost > self._max_stale_frames:
            return None
        return self._homography

    def get_inverse_homography(self) -> np.ndarray | None:
        """Returns table->screen homography (for drawing AR overlays)."""
        if self._inv_homography is None:
            return None
        if self._frames_since_lost > self._max_stale_frames:
            return None
        return self._inv_homography

    def screen_to_table(self, points: np.ndarray) -> np.ndarray | None:
        """Transform Nx2 screen coords to table coords.

        Args:
            points: shape (N, 2) in screen pixel space.

        Returns:
            points shape (N, 2) in table space, or None if no valid homography.
        """
        H = self.get_homography()
        if H is None:
            return None
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, H)
        return transformed.reshape(-1, 2)

    def table_to_screen(self, points: np.ndarray) -> np.ndarray | None:
        """Transform Nx2 table coords to screen coords (for drawing).

        Args:
            points: shape (N, 2) in table coordinate space.

        Returns:
            points shape (N, 2) in screen pixel space, or None if no valid homography.
        """
        H_inv = self.get_inverse_homography()
        if H_inv is None:
            return None
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, H_inv)
        return transformed.reshape(-1, 2)

    def get_screen_corners(self) -> np.ndarray | None:
        """Returns the 4 corner points in screen space (for debug drawing)."""
        return self._screen_corners

    def is_valid(self) -> bool:
        """Returns True if homography is available and not too stale."""
        return (
            self._homography is not None
            and self._frames_since_lost <= self._max_stale_frames
        )

    @property
    def frames_since_lost(self) -> int:
        return self._frames_since_lost

    @property
    def is_stale(self) -> bool:
        """True if using cached homography (markers not currently visible)."""
        return self._frames_since_lost > 0 and self.is_valid()
