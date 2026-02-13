"""State engine for managing procedure progression.

Checks each frame whether the active step's target object is inside
its target zone. Uses debouncing to avoid accidental triggers and
grace periods to handle brief occlusions.
"""

from __future__ import annotations

import math
from typing import Optional

from tgs.config import StateConfig
from tgs.perception.object_detector import Detection, DetectionResult
from tgs.state.procedure import Procedure, Step, StepStatus


class StateEngine:
    """Manages procedure progression based on perception inputs."""

    def __init__(self, procedure: Procedure, config: StateConfig) -> None:
        self._procedure = procedure
        self._config = config
        # Tracks consecutive frames each object class has been missing
        self._object_missing: dict[str, int] = {}

    def update(self, detections: DetectionResult) -> None:
        """Main per-frame update. Call once per frame with latest detections."""
        step = self.get_current_step()
        if step is None:
            return

        detection = self.get_object_for_step(step, detections)

        if detection is None:
            # Object not found — use grace period
            missing = self._object_missing.get(step.object_class, 0) + 1
            self._object_missing[step.object_class] = missing
            if missing >= self._config.lost_object_grace_frames:
                step.debounce_count = 0
            return

        # Object found — reset missing counter
        self._object_missing[step.object_class] = 0

        if self._check_placement(detection, step):
            step.debounce_count += 1
            if step.debounce_count >= self._config.debounce_frames:
                self._advance_step()
        else:
            step.debounce_count = 0

    def get_current_step(self) -> Step | None:
        return self._procedure.get_current_step()

    def get_procedure(self) -> Procedure:
        return self._procedure

    def is_complete(self) -> bool:
        return self._procedure.is_complete()

    def get_object_for_step(
        self, step: Step, detections: DetectionResult
    ) -> Detection | None:
        """Find the detection matching the step's object_class.

        If multiple matches, pick the one closest to the target zone.
        """
        matches = detections.get_by_class(step.object_class)
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]

        # Pick closest to target
        tx, ty = step.target_zone.center
        best = None
        best_dist = float("inf")
        for det in matches:
            if det.center_table is None:
                continue
            dx = det.center_table[0] - tx
            dy = det.center_table[1] - ty
            dist = math.hypot(dx, dy)
            if dist < best_dist:
                best_dist = dist
                best = det
        return best

    def _check_placement(self, detection: Detection, step: Step) -> bool:
        """Returns True if detection's table-space center is within the target zone."""
        if detection.center_table is None:
            return False
        tx, ty = step.target_zone.center
        dx = detection.center_table[0] - tx
        dy = detection.center_table[1] - ty
        dist = math.hypot(dx, dy)
        return dist <= self._config.placement_threshold_px

    def _advance_step(self) -> None:
        """Mark current step COMPLETE, activate next step if any."""
        step = self.get_current_step()
        if step is None:
            return

        step.status = StepStatus.COMPLETE
        step.debounce_count = 0

        # Activate the next pending step
        for s in self._procedure.steps:
            if s.status == StepStatus.PENDING:
                s.status = StepStatus.ACTIVE
                break

    def reset(self) -> None:
        self._procedure.reset()
        self._object_missing.clear()

    def get_progress(self) -> tuple[int, int]:
        """Returns (completed_steps, total_steps)."""
        completed = sum(
            1 for s in self._procedure.steps if s.status == StepStatus.COMPLETE
        )
        return completed, len(self._procedure.steps)

    def get_feedback_for_step(
        self, step: Step, detections: DetectionResult
    ) -> dict:
        """Returns feedback dict for rendering."""
        detection = self.get_object_for_step(step, detections)
        tx, ty = step.target_zone.center

        if detection is None or detection.center_table is None:
            return {
                "object_found": False,
                "object_position": None,
                "target_position": (tx, ty),
                "distance": None,
                "in_zone": False,
                "debounce_progress": 0.0,
            }

        dx = detection.center_table[0] - tx
        dy = detection.center_table[1] - ty
        dist = math.hypot(dx, dy)
        in_zone = dist <= self._config.placement_threshold_px
        debounce_progress = step.debounce_count / max(1, self._config.debounce_frames)

        return {
            "object_found": True,
            "object_position": detection.center_table,
            "target_position": (tx, ty),
            "distance": dist,
            "in_zone": in_zone,
            "debounce_progress": debounce_progress,
        }
