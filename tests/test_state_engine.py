"""Tests for the StateEngine."""

from tgs.config import StateConfig
from tgs.perception.object_detector import Detection, DetectionResult
from tgs.state.engine import StateEngine
from tgs.state.procedure import Procedure, StepStatus, TargetZone, Step


def _make_detection(class_name: str, table_pos: tuple[float, float]) -> Detection:
    """Helper to create a Detection with a table-space position."""
    return Detection(
        class_id=41,
        class_name=class_name,
        confidence=0.9,
        bbox=(0, 0, 50, 50),
        center_screen=(25, 25),
        center_table=table_pos,
    )


def _make_result(*detections: Detection) -> DetectionResult:
    return DetectionResult(detections=list(detections))


def _default_engine() -> StateEngine:
    """Create a StateEngine with a simple 3-step procedure."""
    proc = Procedure(
        name="Test",
        steps=[
            Step(0, "Place cup", "cup",
                 TargetZone((200, 200), 50.0, "A"), StepStatus.ACTIVE),
            Step(1, "Place bottle", "bottle",
                 TargetZone((500, 200), 50.0, "B"), StepStatus.PENDING),
            Step(2, "Place bowl", "bowl",
                 TargetZone((800, 200), 50.0, "C"), StepStatus.PENDING),
        ],
    )
    config = StateConfig(placement_threshold_px=50.0, debounce_frames=5, lost_object_grace_frames=10)
    return StateEngine(proc, config)


def test_step_advances_after_debounce():
    engine = _default_engine()
    det = _make_detection("cup", (200, 200))  # exactly on target

    for _ in range(5):
        engine.update(_make_result(det))

    # Step 0 should be complete, step 1 should be active
    assert engine.get_procedure().steps[0].status == StepStatus.COMPLETE
    assert engine.get_procedure().steps[1].status == StepStatus.ACTIVE
    completed, total = engine.get_progress()
    assert completed == 1
    assert total == 3


def test_debounce_resets_when_object_leaves_zone():
    engine = _default_engine()
    det_in = _make_detection("cup", (200, 200))  # on target
    det_out = _make_detection("cup", (400, 400))  # far away

    # Partially debounce
    for _ in range(3):
        engine.update(_make_result(det_in))
    assert engine.get_procedure().steps[0].debounce_count == 3

    # Object leaves zone
    engine.update(_make_result(det_out))
    assert engine.get_procedure().steps[0].debounce_count == 0

    # Step should NOT have advanced
    assert engine.get_procedure().steps[0].status == StepStatus.ACTIVE


def test_grace_period_preserves_debounce():
    engine = _default_engine()
    det = _make_detection("cup", (200, 200))

    # Build up debounce
    for _ in range(3):
        engine.update(_make_result(det))
    assert engine.get_procedure().steps[0].debounce_count == 3

    # Object disappears for fewer frames than grace period
    empty = _make_result()
    for _ in range(5):
        engine.update(empty)

    # Debounce should be preserved (within grace period of 10)
    assert engine.get_procedure().steps[0].debounce_count == 3


def test_grace_period_expires_resets_debounce():
    engine = _default_engine()
    det = _make_detection("cup", (200, 200))

    for _ in range(3):
        engine.update(_make_result(det))

    # Object disappears for longer than grace period
    empty = _make_result()
    for _ in range(15):
        engine.update(empty)

    assert engine.get_procedure().steps[0].debounce_count == 0


def test_full_procedure_completion():
    engine = _default_engine()

    # Complete all 3 steps
    objects = [
        ("cup", (200, 200)),
        ("bottle", (500, 200)),
        ("bowl", (800, 200)),
    ]
    for obj_class, pos in objects:
        det = _make_detection(obj_class, pos)
        for _ in range(5):
            engine.update(_make_result(det))

    assert engine.is_complete()
    completed, total = engine.get_progress()
    assert completed == 3


def test_multiple_objects_picks_closest():
    engine = _default_engine()
    # Two cups: one close to target, one far
    det_close = _make_detection("cup", (210, 200))  # near target (200, 200)
    det_far = _make_detection("cup", (600, 300))

    result = _make_result(det_close, det_far)
    matched = engine.get_object_for_step(engine.get_current_step(), result)
    assert matched is det_close


def test_feedback_when_object_found():
    engine = _default_engine()
    det = _make_detection("cup", (220, 200))
    result = _make_result(det)

    step = engine.get_current_step()
    feedback = engine.get_feedback_for_step(step, result)
    assert feedback["object_found"] is True
    assert feedback["distance"] is not None
    assert feedback["distance"] == 20.0  # |220-200|


def test_feedback_when_object_missing():
    engine = _default_engine()
    result = _make_result()

    step = engine.get_current_step()
    feedback = engine.get_feedback_for_step(step, result)
    assert feedback["object_found"] is False
    assert feedback["distance"] is None


def test_reset():
    engine = _default_engine()
    det = _make_detection("cup", (200, 200))
    for _ in range(5):
        engine.update(_make_result(det))
    assert engine.get_progress()[0] == 1

    engine.reset()
    assert engine.get_progress()[0] == 0
    assert engine.get_procedure().steps[0].status == StepStatus.ACTIVE
