import pytest

from app.services.motion_analysis import calculate_sequence_risk_scores


ANGLE_SEQUENCE = [
    {"left_elbow": 169.1, "right_elbow": 162.7, "left_shoulder": 0.9,  "right_shoulder": 2.1,  "left_knee": 173.4, "right_knee": 162.2, "left_hip": 163.8, "right_hip": 173.9},
    {"left_elbow": 169.3, "right_elbow": 163.4, "left_shoulder": 0.5,  "right_shoulder": 1.6,  "left_knee": 173.7, "right_knee": 162.4, "left_hip": 164.3, "right_hip": 173.4},
    {"left_elbow": 168.9, "right_elbow": 162.1, "left_shoulder": 1.1,  "right_shoulder": 2.5,  "left_knee": 172.9, "right_knee": 161.8, "left_hip": 163.2, "right_hip": 174.1},
]


class TestCalculateSequenceRiskScores:

    def test_returns_list_with_length_equal_to_detected_sequence(self):
        result = calculate_sequence_risk_scores(
            ANGLE_SEQUENCE,
            ANGLE_SEQUENCE
        )

        assert isinstance(result, list)
        assert len(result) == len(ANGLE_SEQUENCE)

    def test_all_risk_scores_within_valid_range(self):
        result = calculate_sequence_risk_scores(
            ANGLE_SEQUENCE,
            ANGLE_SEQUENCE
        )

        for score in result:
            assert isinstance(score, float)
            assert 0.0 <= score <= 100.0

    def test_identical_sequences_returns_all_zero_risk_scores(self):
        result = calculate_sequence_risk_scores(
            ANGLE_SEQUENCE,
            ANGLE_SEQUENCE
        )

        for score in result:
            assert score < 1e-4, \
                f"Expected score near 0.0, got {score}"