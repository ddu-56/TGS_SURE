"""AR overlay rendering for the TGS application.

Draws all visual feedback onto the camera frame: table boundary, grid,
target zones, guidance arrows, bounding boxes, HUD, and warnings.
Uses the inverse homography (table->screen) to project table-space
drawings onto the camera image.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cv2
import numpy as np

from tgs.viz import colors
from tgs.config import VizConfig

if TYPE_CHECKING:
    from tgs.perception.surface_tracker import SurfaceTracker
    from tgs.perception.object_detector import DetectionResult
    from tgs.state.engine import StateEngine
    from tgs.state.procedure import StepStatus


class Renderer:
    """Draws all AR overlays onto the camera frame."""

    def __init__(self, config: VizConfig) -> None:
        self._config = config

    def draw_all(
        self,
        frame: np.ndarray,
        surface_tracker: SurfaceTracker,
        state_engine: StateEngine,
        detections: DetectionResult,
        fps: float,
    ) -> np.ndarray:
        """Master draw method. Composes all overlays onto frame."""
        if surface_tracker.is_valid():
            self.draw_table_boundary(frame, surface_tracker)
            self.draw_table_grid(frame, surface_tracker)
            self.draw_target_zones(frame, surface_tracker, state_engine)
            self.draw_guidance_arrow(frame, surface_tracker, state_engine, detections)
            self.draw_detection_boxes(frame, detections)

            if surface_tracker.is_stale:
                self._draw_stale_warning(frame, surface_tracker.frames_since_lost)
        else:
            self.draw_lost_marker_warning(frame)

        self.draw_hud(frame, state_engine, fps)

        if state_engine.is_complete():
            self.draw_completion_screen(frame)

        return frame

    def draw_table_boundary(
        self, frame: np.ndarray, surface_tracker: SurfaceTracker
    ) -> None:
        """Draw the table boundary quadrilateral from the 4 corner markers."""
        screen_corners = surface_tracker.get_screen_corners()
        if screen_corners is None:
            return
        pts = screen_corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], isClosed=True, color=colors.GREEN, thickness=2)

    def draw_table_grid(
        self, frame: np.ndarray, surface_tracker: SurfaceTracker
    ) -> None:
        """Draw an NxN grid on the table surface using perspective transform."""
        cfg = self._config
        tw = surface_tracker._calibration.table_width
        th = surface_tracker._calibration.table_height
        n = cfg.grid_lines

        for i in range(1, n):
            # Horizontal line
            y = th * i / n
            h_pts_table = np.array([[0, y], [tw, y]], dtype=np.float32)
            h_pts_screen = surface_tracker.table_to_screen(h_pts_table)
            if h_pts_screen is not None:
                p1 = tuple(h_pts_screen[0].astype(int))
                p2 = tuple(h_pts_screen[1].astype(int))
                cv2.line(frame, p1, p2, cfg.color_grid, 1)

            # Vertical line
            x = tw * i / n
            v_pts_table = np.array([[x, 0], [x, th]], dtype=np.float32)
            v_pts_screen = surface_tracker.table_to_screen(v_pts_table)
            if v_pts_screen is not None:
                p1 = tuple(v_pts_screen[0].astype(int))
                p2 = tuple(v_pts_screen[1].astype(int))
                cv2.line(frame, p1, p2, cfg.color_grid, 1)

    def draw_target_zones(
        self,
        frame: np.ndarray,
        surface_tracker: SurfaceTracker,
        state_engine: StateEngine,
    ) -> None:
        """Draw target zone circles for all steps.

        COMPLETE zones: green filled semi-transparent
        ACTIVE zone: yellow pulsing
        PENDING zones: dim gray outline
        """
        from tgs.state.procedure import StepStatus

        procedure = state_engine.get_procedure()
        for step in procedure.steps:
            zone = step.target_zone
            screen_poly = self._project_circle_to_screen(
                zone.center, zone.radius, surface_tracker
            )
            if screen_poly is None:
                continue

            pts = screen_poly.astype(np.int32).reshape(-1, 1, 2)

            if step.status == StepStatus.COMPLETE:
                # Semi-transparent green fill
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], self._config.color_correct)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                cv2.polylines(frame, [pts], True, self._config.color_correct, 2)
            elif step.status == StepStatus.ACTIVE:
                cv2.polylines(frame, [pts], True, self._config.color_pending, self._config.zone_thickness)
                # Draw label
                center_screen = surface_tracker.table_to_screen(
                    np.array([zone.center], dtype=np.float32)
                )
                if center_screen is not None:
                    pos = tuple(center_screen[0].astype(int))
                    cv2.putText(
                        frame, zone.label, (pos[0] - 20, pos[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self._config.color_pending, 2,
                    )
            else:
                cv2.polylines(frame, [pts], True, colors.DARK_GRAY, 1)

    def draw_guidance_arrow(
        self,
        frame: np.ndarray,
        surface_tracker: SurfaceTracker,
        state_engine: StateEngine,
        detections: DetectionResult,
    ) -> None:
        """Draw an arrow from the active object to its target zone."""
        step = state_engine.get_current_step()
        if step is None:
            return

        feedback = state_engine.get_feedback_for_step(step, detections)
        if not feedback["object_found"] or feedback["object_position"] is None:
            return

        obj_table = np.array([feedback["object_position"]], dtype=np.float32)
        tgt_table = np.array([feedback["target_position"]], dtype=np.float32)

        obj_screen = surface_tracker.table_to_screen(obj_table)
        tgt_screen = surface_tracker.table_to_screen(tgt_table)
        if obj_screen is None or tgt_screen is None:
            return

        p1 = tuple(obj_screen[0].astype(int))
        p2 = tuple(tgt_screen[0].astype(int))

        # Color interpolation: red (far) -> green (close)
        dist = feedback["distance"] or 0
        threshold = step.target_zone.radius * 3
        t = max(0.0, min(1.0, 1.0 - dist / threshold))
        color = self._lerp_color(self._config.color_incorrect, self._config.color_correct, t)

        cv2.arrowedLine(frame, p1, p2, color, self._config.arrow_thickness, tipLength=0.15)

    def draw_detection_boxes(
        self, frame: np.ndarray, detections: DetectionResult
    ) -> None:
        """Draw bounding boxes and labels for all detected objects."""
        for det in detections.detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), colors.LIGHT_BLUE, 2)
            label = f"{det.class_name} {det.confidence:.0%}"
            cv2.putText(
                frame, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.LIGHT_BLUE, 1,
            )

    def draw_hud(
        self,
        frame: np.ndarray,
        state_engine: StateEngine,
        fps: float,
    ) -> None:
        """Draw the heads-up display: instruction banner, step counter, FPS."""
        h, w = frame.shape[:2]
        cfg = self._config
        banner_h = cfg.hud_banner_height

        # Semi-transparent banner at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), colors.BLACK, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        step = state_engine.get_current_step()
        completed, total = state_engine.get_progress()

        if state_engine.is_complete():
            instruction = "All steps complete!"
        elif step is not None:
            instruction = step.instruction
        else:
            instruction = "Waiting..."

        # Instruction text
        cv2.putText(
            frame, instruction, (15, banner_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale, colors.WHITE, cfg.font_thickness,
        )

        # Step counter on the right
        progress_text = f"Step {completed}/{total}"
        text_size = cv2.getTextSize(
            progress_text, cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale, cfg.font_thickness
        )[0]
        cv2.putText(
            frame, progress_text, (w - text_size[0] - 15, banner_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale, colors.GREEN, cfg.font_thickness,
        )

        # FPS counter (debug)
        if cfg.show_debug_info:
            cv2.putText(
                frame, f"FPS: {fps:.0f}", (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.GREEN, 1,
            )

    def draw_lost_marker_warning(self, frame: np.ndarray) -> None:
        """Draw a full warning overlay when ArUco markers are not detected."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), colors.RED, -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        msg = "ArUco markers not detected"
        sub = "Place 4 markers at table corners"
        cv2.putText(
            frame, msg, (w // 2 - 250, h // 2 - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, colors.RED, 2,
        )
        cv2.putText(
            frame, sub, (w // 2 - 220, h // 2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors.WHITE, 1,
        )

    def draw_completion_screen(self, frame: np.ndarray) -> None:
        """Draw the 'Task Complete!' celebration overlay."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), colors.GREEN, -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        msg = "Task Complete!"
        text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)[0]
        x = (w - text_size[0]) // 2
        y = (h + text_size[1]) // 2
        cv2.putText(frame, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, colors.GREEN, 3)

        sub = "Press 'r' to restart or 'q' to quit"
        sub_size = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
        cv2.putText(
            frame, sub, ((w - sub_size[0]) // 2, y + 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors.WHITE, 1,
        )

    def _draw_stale_warning(self, frame: np.ndarray, frames_lost: int) -> None:
        """Draw a subtle warning when using cached (stale) homography."""
        alpha = min(1.0, frames_lost / 60.0)  # fade in over 2 seconds
        color = self._lerp_color(colors.YELLOW, colors.ORANGE, alpha)
        cv2.putText(
            frame, "Markers lost - using cached position",
            (15, frame.shape[0] - 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
        )

    def _project_circle_to_screen(
        self,
        center_table: tuple[float, float],
        radius_table: float,
        surface_tracker: SurfaceTracker,
        num_points: int = 32,
    ) -> np.ndarray | None:
        """Sample a circle in table-space as a polygon, transform to screen-space."""
        angles = np.linspace(0, 2 * math.pi, num_points, endpoint=False)
        cx, cy = center_table
        circle_pts = np.array(
            [[cx + radius_table * math.cos(a), cy + radius_table * math.sin(a)] for a in angles],
            dtype=np.float32,
        )
        return surface_tracker.table_to_screen(circle_pts)

    @staticmethod
    def _lerp_color(
        color_a: tuple[int, int, int],
        color_b: tuple[int, int, int],
        t: float,
    ) -> tuple[int, int, int]:
        """Linearly interpolate between two BGR colors."""
        t = max(0.0, min(1.0, t))
        return tuple(int(a + (b - a) * t) for a, b in zip(color_a, color_b))
