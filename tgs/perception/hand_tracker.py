"""Optional MediaPipe Hands wrapper for hand tracking.

Disabled by default. When enabled, detects hand landmarks in the camera
frame. Can be toggled at runtime via the 'm' key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class HandLandmarks:
    """Landmarks for a single detected hand."""

    landmarks: np.ndarray  # shape (21, 3) normalized
    screen_landmarks: np.ndarray  # shape (21, 2) pixel coords
    handedness: str  # "Left" or "Right"


class HandTracker:
    """MediaPipe Hands wrapper. Optional component — disabled by default."""

    def __init__(self, enabled: bool = False, max_hands: int = 1) -> None:
        self._enabled = enabled
        self._max_hands = max_hands
        self._hands = None
        if enabled:
            self._initialize()

    def _initialize(self) -> None:
        """Lazy init of MediaPipe Hands to avoid import cost when disabled."""
        try:
            import mediapipe as mp

            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=self._max_hands,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except ImportError:
            print("[HandTracker] mediapipe not installed. Disabling hand tracking.")
            self._enabled = False

    def detect(self, frame_rgb: np.ndarray) -> List[HandLandmarks]:
        """Detect hands in an RGB frame. Returns empty list if disabled."""
        if not self._enabled or self._hands is None:
            return []

        h, w = frame_rgb.shape[:2]
        results = self._hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return []

        hands: list[HandLandmarks] = []
        for i, hand_lms in enumerate(results.multi_hand_landmarks):
            landmarks = np.array(
                [(lm.x, lm.y, lm.z) for lm in hand_lms.landmark],
                dtype=np.float32,
            )
            screen_lms = np.array(
                [(lm.x * w, lm.y * h) for lm in hand_lms.landmark],
                dtype=np.float32,
            )
            handedness = "Unknown"
            if results.multi_handedness and i < len(results.multi_handedness):
                handedness = results.multi_handedness[i].classification[0].label

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
