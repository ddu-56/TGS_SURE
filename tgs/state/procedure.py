"""Data classes for task procedures and steps.

A Procedure is an ordered sequence of Steps. Each Step asks the user to
place a specific object in a target zone on the table surface.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple


class StepStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETE = "complete"


@dataclass
class TargetZone:
    """A target region on the table surface, in table coordinates."""

    center: Tuple[float, float]  # (x, y) in table-space
    radius: float  # acceptance radius in table-space pixels
    label: str = ""


@dataclass
class Step:
    """A single step in the sorting procedure."""

    step_index: int
    instruction: str  # e.g., "Move the Cup to Zone A"
    object_class: str  # "cup", "bottle", or "bowl"
    target_zone: TargetZone
    status: StepStatus = StepStatus.PENDING
    debounce_count: int = 0


@dataclass
class Procedure:
    """An ordered sequence of steps defining the full task."""

    name: str
    steps: List[Step] = field(default_factory=list)

    @classmethod
    def create_default_sorting_task(
        cls,
        table_width: float,
        table_height: float,
        zone_radius: float = 60.0,
    ) -> Procedure:
        """Create a default 3-step sorting task with evenly spaced zones."""
        margin_x = table_width * 0.2
        margin_y = table_height * 0.5  # center vertically
        spacing = (table_width - 2 * margin_x) / 2

        objects = [
            ("cup", "Zone A"),
            ("bottle", "Zone B"),
            ("bowl", "Zone C"),
        ]

        steps = []
        for i, (obj, label) in enumerate(objects):
            cx = margin_x + spacing * i
            cy = margin_y
            steps.append(
                Step(
                    step_index=i,
                    instruction=f"Place the {obj.title()} in {label}",
                    object_class=obj,
                    target_zone=TargetZone(center=(cx, cy), radius=zone_radius, label=label),
                )
            )

        proc = cls(name="Default Sorting Task", steps=steps)
        proc.steps[0].status = StepStatus.ACTIVE
        return proc

    @classmethod
    def create_randomized_task(
        cls,
        table_width: float,
        table_height: float,
        zone_radius: float = 60.0,
        margin: float = 80.0,
    ) -> Procedure:
        """Create a task with randomized target positions within the table bounds."""
        objects = [
            ("cup", "Zone A"),
            ("bottle", "Zone B"),
            ("bowl", "Zone C"),
        ]

        steps = []
        placed_centers: list[tuple[float, float]] = []

        for i, (obj, label) in enumerate(objects):
            # Try to place zones without overlap
            for _ in range(100):
                cx = random.uniform(margin, table_width - margin)
                cy = random.uniform(margin, table_height - margin)
                # Check minimum distance from existing zones
                too_close = any(
                    ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 < zone_radius * 3
                    for px, py in placed_centers
                )
                if not too_close:
                    break

            placed_centers.append((cx, cy))
            steps.append(
                Step(
                    step_index=i,
                    instruction=f"Place the {obj.title()} in {label}",
                    object_class=obj,
                    target_zone=TargetZone(center=(cx, cy), radius=zone_radius, label=label),
                )
            )

        proc = cls(name="Randomized Sorting Task", steps=steps)
        proc.steps[0].status = StepStatus.ACTIVE
        return proc

    def get_current_step(self) -> Step | None:
        """Returns the first ACTIVE step, or first PENDING if none active."""
        for step in self.steps:
            if step.status == StepStatus.ACTIVE:
                return step
        for step in self.steps:
            if step.status == StepStatus.PENDING:
                return step
        return None

    def is_complete(self) -> bool:
        return all(s.status == StepStatus.COMPLETE for s in self.steps)

    def reset(self) -> None:
        for step in self.steps:
            step.status = StepStatus.PENDING
            step.debounce_count = 0
        if self.steps:
            self.steps[0].status = StepStatus.ACTIVE
