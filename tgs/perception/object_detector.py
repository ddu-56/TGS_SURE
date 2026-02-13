"""Threaded YOLOv8 object detection wrapper.

Runs YOLOv8n inference on a dedicated worker thread so the main render loop
is never blocked by inference latency. Uses latest-frame semantics: only the
most recent submitted frame is processed; intermediate frames are silently
dropped.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from tgs.config import YoloConfig, PerformanceConfig


@dataclass
class Detection:
    """A single object detection result."""

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 screen space
    center_screen: tuple[int, int]
    center_table: Optional[tuple[float, float]] = None
    timestamp: float = 0.0


@dataclass
class DetectionResult:
    """All detections from a single inference pass."""

    detections: List[Detection] = field(default_factory=list)
    timestamp: float = 0.0
    inference_time_ms: float = 0.0

    def get_by_class(self, class_name: str) -> List[Detection]:
        """Return all detections matching a class name."""
        return [d for d in self.detections if d.class_name == class_name]


class ObjectDetector:
    """Runs YOLOv8 inference on a worker thread.

    Main thread feeds frames via submit_frame() (non-blocking).
    Main thread reads results via get_latest_detections() (non-blocking).
    Worker processes the latest submitted frame continuously.
    """

    def __init__(
        self,
        config: YoloConfig,
        performance_config: PerformanceConfig,
    ) -> None:
        self._config = config
        self._perf_config = performance_config
        self._model = None

        # Thread-safe frame exchange
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_ready = threading.Event()

        # Thread-safe result storage
        self._result_lock = threading.Lock()
        self._latest_result: DetectionResult = DetectionResult()

        self._worker_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Load the YOLO model and start the worker thread."""
        from ultralytics import YOLO

        self._model = YOLO(self._config.model_path)
        self._running = True

        if self._perf_config.threading_enabled:
            self._worker_thread = threading.Thread(
                target=self._worker_loop, daemon=True, name="yolo-worker"
            )
            self._worker_thread.start()

    def stop(self) -> None:
        """Signal worker thread to stop and wait for it."""
        self._running = False
        self._frame_ready.set()  # wake up worker so it can exit
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=3.0)
            self._worker_thread = None

    def submit_frame(self, frame: np.ndarray) -> None:
        """Submit a new frame for inference. Non-blocking.

        If threading is disabled, runs inference synchronously.
        """
        if not self._perf_config.threading_enabled:
            result = self._run_inference(frame)
            with self._result_lock:
                self._latest_result = result
            return

        with self._frame_lock:
            self._latest_frame = frame.copy()
        self._frame_ready.set()

    def get_latest_detections(self) -> DetectionResult:
        """Get the most recent detection results. Non-blocking."""
        with self._result_lock:
            return self._latest_result

    def _worker_loop(self) -> None:
        """Worker thread main loop."""
        while self._running:
            try:
                self._frame_ready.wait(timeout=0.1)
                self._frame_ready.clear()

                with self._frame_lock:
                    frame = self._latest_frame
                    self._latest_frame = None

                if frame is None:
                    continue

                result = self._run_inference(frame)
                with self._result_lock:
                    self._latest_result = result

            except Exception as e:
                print(f"[ObjectDetector] Worker error: {e}")
                continue

    def _run_inference(self, frame: np.ndarray) -> DetectionResult:
        """Run YOLO inference on a single frame. Returns filtered detections."""
        if self._model is None:
            return DetectionResult()

        t0 = time.perf_counter()
        target_ids = list(self._config.target_class_ids.keys())
        results = self._model.predict(
            frame,
            conf=self._config.confidence_threshold,
            iou=self._config.iou_threshold,
            classes=target_ids,
            verbose=False,
            imgsz=self._config.input_size,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        now = time.time()

        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].int().tolist()
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                class_name = self._config.target_class_ids.get(cls_id, f"class_{cls_id}")
                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=class_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        center_screen=(cx, cy),
                        timestamp=now,
                    )
                )

        return DetectionResult(
            detections=detections,
            timestamp=now,
            inference_time_ms=elapsed_ms,
        )

    @staticmethod
    def map_to_table(
        detections: DetectionResult,
        surface_tracker,
    ) -> None:
        """Enrich detections with table-space coordinates using homography.

        Modifies detections in-place.
        """
        for det in detections.detections:
            pts = np.array([det.center_screen], dtype=np.float32)
            table_pts = surface_tracker.screen_to_table(pts)
            if table_pts is not None:
                det.center_table = (float(table_pts[0][0]), float(table_pts[0][1]))
