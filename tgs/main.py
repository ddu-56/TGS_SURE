"""Main application entry point for the Physical Task Guidance System.

Orchestrates the full pipeline: camera capture, perception, state
management, visualization, and display. Handles keyboard input and
component lifecycle.
"""

from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

from tgs.config import TGSConfig
from tgs.perception.surface_tracker import SurfaceTracker
from tgs.perception.object_detector import ObjectDetector, DetectionResult
from tgs.perception.hand_tracker import HandTracker
from tgs.state.procedure import Procedure
from tgs.state.engine import StateEngine
from tgs.viz.renderer import Renderer


class _FPSCounter:
    """Rolling average FPS counter."""

    def __init__(self, window_size: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=window_size)

    def tick(self) -> float:
        now = time.perf_counter()
        self._timestamps.append(now)
        return self.get_fps()

    def get_fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed


class TGSApp:
    """Main application class orchestrating the full pipeline."""

    def __init__(self, config: TGSConfig | None = None) -> None:
        self._config = config or TGSConfig()
        self._cap: cv2.VideoCapture | None = None
        self._surface_tracker: SurfaceTracker | None = None
        self._object_detector: ObjectDetector | None = None
        self._hand_tracker: HandTracker | None = None
        self._state_engine: StateEngine | None = None
        self._renderer: Renderer | None = None
        self._fps = _FPSCounter()
        self._running = False
        self._reconnect_attempts = 0

    def initialize(self) -> bool:
        """Initialize all components. Returns True on success."""
        cfg = self._config

        # Camera
        self._cap = cv2.VideoCapture(cfg.camera.index)
        if not self._cap.isOpened():
            print(f"[TGS] Failed to open camera at index {cfg.camera.index}")
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
        self._cap.set(cv2.CAP_PROP_FPS, cfg.camera.fps)

        # Try to load saved calibration
        cfg.calibration.load()

        # Perception
        self._surface_tracker = SurfaceTracker(cfg.aruco, cfg.calibration)
        self._object_detector = ObjectDetector(cfg.yolo, cfg.performance)
        self._hand_tracker = HandTracker(enabled=cfg.performance.mediapipe_enabled)

        print("[TGS] Loading YOLO model...")
        self._object_detector.start()

        # State
        procedure = Procedure.create_default_sorting_task(
            cfg.calibration.table_width,
            cfg.calibration.table_height,
            cfg.viz.zone_radius,
        )
        self._state_engine = StateEngine(procedure, cfg.state)

        # Visualization
        self._renderer = Renderer(cfg.viz)

        print("[TGS] Initialization complete. Press 'q' to quit.")
        print("[TGS] Keys: r=reset, c=recalibrate, d=toggle debug, m=toggle MediaPipe")
        return True

    def run(self) -> None:
        """Main loop. Runs until 'q' pressed or window closed."""
        self._running = True

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                self._reconnect_attempts += 1
                if self._reconnect_attempts > 30:
                    print("[TGS] Camera disconnected. Attempting to reconnect...")
                    self._cap.release()
                    time.sleep(1.0)
                    self._cap = cv2.VideoCapture(self._config.camera.index)
                    self._reconnect_attempts = 0
                continue
            self._reconnect_attempts = 0

            annotated = self._process_frame(frame)

            cv2.imshow("TGS - Physical Task Guidance System", annotated)
            key = cv2.waitKey(1) & 0xFF
            if not self._handle_input(key):
                break

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through the full pipeline."""
        # 1. Surface tracking (fast, main thread)
        self._surface_tracker.detect(frame)

        # 2. Submit frame to YOLO worker (non-blocking)
        self._object_detector.submit_frame(frame)

        # 3. Read latest YOLO results
        detections = self._object_detector.get_latest_detections()

        # 4. Map detections to table space
        if self._surface_tracker.is_valid():
            ObjectDetector.map_to_table(detections, self._surface_tracker)

        # 5. Optional hand tracking
        if self._hand_tracker.enabled:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands = self._hand_tracker.detect(frame_rgb)
            # Draw hand landmarks
            for hand in hands:
                for pt in hand.screen_landmarks:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (0, 200, 255), -1)

        # 6. Update state engine
        if self._surface_tracker.is_valid():
            self._state_engine.update(detections)

        # 7. Render overlays
        fps = self._fps.tick()
        self._renderer.draw_all(
            frame, self._surface_tracker, self._state_engine, detections, fps
        )

        return frame

    def _handle_input(self, key: int) -> bool:
        """Handle keyboard input. Returns False to quit."""
        if key == ord("q") or key == 27:  # q or ESC
            return False
        elif key == ord("r"):
            self._state_engine.reset()
            print("[TGS] Procedure reset.")
        elif key == ord("c"):
            # Force recalibration by invalidating cached homography
            self._config.calibration.calibrated = False
            self._config.calibration.homography_matrix = None
            self._surface_tracker = SurfaceTracker(
                self._config.aruco, self._config.calibration
            )
            print("[TGS] Recalibration triggered.")
        elif key == ord("d"):
            self._config.viz.show_debug_info = not self._config.viz.show_debug_info
            state = "on" if self._config.viz.show_debug_info else "off"
            print(f"[TGS] Debug info: {state}")
        elif key == ord("m"):
            enabled = not self._hand_tracker.enabled
            self._hand_tracker = HandTracker(enabled=enabled)
            state = "on" if enabled else "off"
            print(f"[TGS] MediaPipe hand tracking: {state}")
        elif key == ord("n"):
            # Generate new randomized task
            procedure = Procedure.create_randomized_task(
                self._config.calibration.table_width,
                self._config.calibration.table_height,
                self._config.viz.zone_radius,
            )
            self._state_engine = StateEngine(procedure, self._config.state)
            print("[TGS] New randomized task generated.")
        return True

    def cleanup(self) -> None:
        """Release camera, stop worker threads, destroy windows."""
        if self._object_detector:
            self._object_detector.stop()
        if self._cap:
            self._cap.release()
        cv2.destroyAllWindows()

        # Save calibration for next session
        if self._config.calibration.calibrated:
            self._config.calibration.save()
            print("[TGS] Calibration saved.")
        print("[TGS] Cleanup complete.")


def main() -> None:
    """Entry point."""
    config = TGSConfig()
    app = TGSApp(config)
    if app.initialize():
        try:
            app.run()
        finally:
            app.cleanup()


if __name__ == "__main__":
    main()
