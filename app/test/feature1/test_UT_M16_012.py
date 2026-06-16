import pytest

from app.services.motion_analysis import validate_exercise_match
from app.exceptions import ExerciseMismatchException


# ANGLE_SEQUENCE (Appendix L) — Squat sequence with sufficient lower_body_range
ANGLE_SEQUENCE = [
    {"left_elbow": 169.1, "right_elbow": 162.7, "left_shoulder": 0.9,  "right_shoulder": 2.1,  "left_knee": 173.4, "right_knee": 162.2, "left_hip": 163.8, "right_hip": 173.9},
    {"left_elbow": 169.3, "right_elbow": 163.4, "left_shoulder": 0.5,  "right_shoulder": 1.6,  "left_knee": 150.0, "right_knee": 148.0, "left_hip": 140.0, "right_hip": 141.0},
    {"left_elbow": 168.9, "right_elbow": 162.1, "left_shoulder": 1.1,  "right_shoulder": 2.5,  "left_knee": 172.9, "right_knee": 161.8, "left_hip": 163.2, "right_hip": 174.1},
]

# Sequence ที่ lower_body_range น้อยกว่า 12.0 → Squat rule fails
SQUAT_FAIL_SEQUENCE = [
    {"left_elbow": 169.1, "right_elbow": 162.7, "left_shoulder": 0.9, "right_shoulder": 2.1, "left_knee": 173.4, "right_knee": 172.9, "left_hip": 163.8, "right_hip": 163.2},
    {"left_elbow": 169.3, "right_elbow": 163.4, "left_shoulder": 0.5, "right_shoulder": 1.6, "left_knee": 174.0, "right_knee": 173.5, "left_hip": 164.0, "right_hip": 163.8},
    {"left_elbow": 168.9, "right_elbow": 162.1, "left_shoulder": 1.1, "right_shoulder": 2.5, "left_knee": 173.8, "right_knee": 173.2, "left_hip": 163.5, "right_hip": 163.1},
]

# Sequence ที่ต่างจาก reference มากเกินไป → low similarity
LOW_SIMILARITY_SEQUENCE = [
    {"left_elbow": 10.0, "right_elbow": 10.0, "left_shoulder": 170.0, "right_shoulder": 170.0, "left_knee": 10.0, "right_knee": 10.0, "left_hip": 10.0, "right_hip": 10.0},
    {"left_elbow": 10.0, "right_elbow": 10.0, "left_shoulder": 170.0, "right_shoulder": 170.0, "left_knee": 10.0, "right_knee": 10.0, "left_hip": 10.0, "right_hip": 10.0},
    {"left_elbow": 10.0, "right_elbow": 10.0, "left_shoulder": 170.0, "right_shoulder": 170.0, "left_knee": 10.0, "right_knee": 10.0, "left_hip": 10.0, "right_hip": 10.0},
]


class TestValidateExerciseMatch:

    def test_matching_sequence_returns_comparison_with_rule_check(self):
        result = validate_exercise_match(
            ANGLE_SEQUENCE,
            ANGLE_SEQUENCE,
            "Squat_men"
        )

        assert "similarity_score"     in result
        assert "average_angle_error"  in result
        assert "movement_range_error" in result
        assert "phase_shift"          in result
        assert "compared_joints"      in result
        assert "rule_check"           in result
        assert result["rule_check"]["passed"] is True

    def test_low_similarity_raises_exercise_mismatch_exception(self):
        with pytest.raises(ExerciseMismatchException) as exc_info:
            validate_exercise_match(
                LOW_SIMILARITY_SEQUENCE,
                ANGLE_SEQUENCE,
                "Squat_men"
            )

        error = exc_info.value
        assert hasattr(error, "similarity_score")
        assert isinstance(float(error.similarity_score), float)
        assert "average_angle_error"  in error.details
        assert "movement_range_error" in error.details
        assert "rule_check"           in error.details

    def test_squat_rule_fail_raises_exercise_mismatch_exception(self):
        with pytest.raises(ExerciseMismatchException) as exc_info:
            validate_exercise_match(
                SQUAT_FAIL_SEQUENCE,
                ANGLE_SEQUENCE,
                "Squat_men"
            )

        error = exc_info.value
        assert error.details["rule_check"]["passed"] is False
        assert error.details["rule_check"]["reason"] == "squat_lower_body_motion"