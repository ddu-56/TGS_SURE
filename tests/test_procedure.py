"""Tests for Procedure and Step data classes."""

from tgs.state.procedure import Procedure, Step, StepStatus, TargetZone


def test_create_default_sorting_task():
    proc = Procedure.create_default_sorting_task(1000.0, 440.0)
    assert len(proc.steps) == 3
    assert proc.steps[0].object_class == "cup"
    assert proc.steps[1].object_class == "bottle"
    assert proc.steps[2].object_class == "bowl"
    # First step should be ACTIVE
    assert proc.steps[0].status == StepStatus.ACTIVE
    assert proc.steps[1].status == StepStatus.PENDING
    assert proc.steps[2].status == StepStatus.PENDING


def test_create_randomized_task():
    proc = Procedure.create_randomized_task(1000.0, 440.0, margin=80.0)
    assert len(proc.steps) == 3
    # All zones should be within bounds
    for step in proc.steps:
        cx, cy = step.target_zone.center
        assert 80.0 <= cx <= 920.0
        assert 80.0 <= cy <= 360.0


def test_get_current_step():
    proc = Procedure.create_default_sorting_task(1000.0, 440.0)
    step = proc.get_current_step()
    assert step is not None
    assert step.step_index == 0
    assert step.object_class == "cup"


def test_is_complete():
    proc = Procedure.create_default_sorting_task(1000.0, 440.0)
    assert not proc.is_complete()
    for step in proc.steps:
        step.status = StepStatus.COMPLETE
    assert proc.is_complete()


def test_reset():
    proc = Procedure.create_default_sorting_task(1000.0, 440.0)
    proc.steps[0].status = StepStatus.COMPLETE
    proc.steps[0].debounce_count = 10
    proc.steps[1].status = StepStatus.ACTIVE
    proc.reset()
    assert proc.steps[0].status == StepStatus.ACTIVE
    assert proc.steps[0].debounce_count == 0
    assert proc.steps[1].status == StepStatus.PENDING


def test_zones_have_labels():
    proc = Procedure.create_default_sorting_task(1000.0, 440.0)
    labels = [s.target_zone.label for s in proc.steps]
    assert labels == ["Zone A", "Zone B", "Zone C"]
