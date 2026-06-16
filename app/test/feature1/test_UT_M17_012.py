import pytest
from unittest.mock import patch

from app.services.motion_analysis import compare_exercise_sequences
from app.exceptions import InvalidKeypointsException


ANGLE_SEQUENCE = [
    {
        "left_elbow": 169.1, "right_elbow": 162.7,
        "left_shoulder": 0.9, "right_shoulder": 2.1,
        "left_knee": 173.4, "right_knee": 162.2,
        "left_hip": 163.8, "right_hip": 173.9
    },
    {
        "left_elbow": 169.3, "right_elbow": 163.4,
        "left_shoulder": 0.5, "right_shoulder": 1.6,
        "left_knee": 173.7, "right_knee": 162.4,
        "left_hip": 164.3, "right_hip": 173.4
    },
    {
        "left_elbow": 168.9, "right_elbow": 162.1,
        "left_shoulder": 1.1, "right_shoulder": 2.5,
        "left_knee": 172.9, "right_knee": 161.8,
        "left_hip": 163.2, "right_hip": 174.1
    },
]


class TestCompareExerciseSequences:

    def test_identical_sequences_returns_correct_structure(self):
        result = compare_exercise_sequences(ANGLE_SEQUENCE, ANGLE_SEQUENCE)

        assert "similarity_score"    in result
        assert "average_angle_error" in result
        assert "movement_range_error" in result
        assert "phase_shift"         in result
        assert "compared_joints"     in result

        assert result["similarity_score"]     >= 99.0
        assert result["average_angle_error"]  < 1e-4
        assert result["movement_range_error"] < 1e-4
        assert isinstance(result["phase_shift"],      int)
        assert isinstance(result["compared_joints"],  list)

    def test_empty_detected_sequence_raises_exception(self):
        with pytest.raises(InvalidKeypointsException) as exc_info:
            compare_exercise_sequences([], ANGLE_SEQUENCE)

        assert "No body joints could be reliably identified" in str(exc_info.value)

    def test_empty_reference_sequence_raises_exception(self):
        with pytest.raises(InvalidKeypointsException) as exc_info:
            compare_exercise_sequences(ANGLE_SEQUENCE, [])

        assert "No body joints could be reliably identified" in str(exc_info.value)

    def test_no_common_joints_raises_exception(self):
        detected   = [{"left_knee": 170.0}]
        reference  = [{"right_hip": 160.0}]

        with pytest.raises(InvalidKeypointsException) as exc_info:
            compare_exercise_sequences(detected, reference)

        assert "No body joints could be reliably identified" in str(exc_info.value)