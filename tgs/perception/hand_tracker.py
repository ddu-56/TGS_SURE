"""Optional MediaPipe Hands wrapper for hand tracking.

Disabled by default. When enabled, detects hand landmarks in the camera
frame using the MediaPipe Tasks API (0.10+). Can be toggled at runtime
via the 'm' key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class HandLandmarks:
    """Landmarks for a single detected hand."""

    landmarks: np.ndarray  # shape (21, 3) normalized
    screen_landmarks: np.ndarray  # shape (21, 2) pixel coords
    handedness: str  # "Left" or "Right"


# Model filename expected next to this module or in working directory
_MODEL_FILENAME = "hand_landmarker.task"
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


class HandTracker:
    """MediaPipe Hands wrapper. Optional component — disabled by default."""

    def __init__(self, enabled: bool = False, max_hands: int = 1) -> None:
        self._enabled = enabled
        self._max_hands = max_hands
        self._landmarker = None
        if enabled:
            self._initialize()

    def _initialize(self) -> None:
        """Init MediaPipe HandLandmarker (Tasks API 0.10+)."""
        try:
            import mediapipe as mp

            model_path = self._find_or_download_model()
            if model_path is None:
                print("[HandTracker] Could not obtain model. Disabling.")
                self._enabled = False
                return

            BaseOptions = mp.tasks.BaseOptions
            HandLandmarkerClass = mp.tasks.vision.HandLandmarker
            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.IMAGE,
                num_hands=self._max_hands,
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = HandLandmarkerClass.create_from_options(options)
        except (ImportError, Exception) as e:
            print(f"[HandTracker] Initialization failed: {e}. Disabling.")
            self._enabled = False

    @staticmethod
    def _find_or_download_model() -> str | None:
        """Find the model file locally, or download it."""
        # Check common locations
        for path in [_MODEL_FILENAME, os.path.join("models", _MODEL_FILENAME)]:
            if os.path.exists(path):
                return path

        # Download
        try:
            import urllib.request

            os.makedirs("models", exist_ok=True)
            dest = os.path.join("models", _MODEL_FILENAME)
            print(f"[HandTracker] Downloading model to {dest}...")
            urllib.request.urlretrieve(_MODEL_URL, dest)
            return dest
        except Exception as e:
            print(f"[HandTracker] Download failed: {e}")
            return None

    def detect(self, frame_rgb: np.ndarray) -> List[HandLandmarks]:
        """Detect hands in an RGB frame. Returns empty list if disabled."""
        if not self._enabled or self._landmarker is None:
            return []

        import mediapipe as mp

        h, w = frame_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return []

        hands: list[HandLandmarks] = []
        for i, hand_lms in enumerate(result.hand_landmarks):
            landmarks = np.array(
                [(lm.x, lm.y, lm.z) for lm in hand_lms],
                dtype=np.float32,
            )
            screen_lms = np.array(
                [(lm.x * w, lm.y * h) for lm in hand_lms],
                dtype=np.float32,
            )
            handedness = "Unknown"
            if result.handedness and i < len(result.handedness):
                handedness = result.handedness[i][0].category_name

            hands.append(
                HandLandmarks(
                    landmarks=landmarks,
                    screen_landmarks=screen_lms,
                    handedness=handedness,
                )
            )

        return hands

    def get_fingertip_position(self, hand: HandLandmarks) -> tuple[int, int]:
        """Extract index finger tip position (landmark 8) in screen coords."""
        pt = hand.screen_landmarks[8]
        return (int(pt[0]), int(pt[1]))

    @property
    def enabled(self) -> bool:
        return self._enabled
