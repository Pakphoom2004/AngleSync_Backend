import pytest

from app.services.motion_analysis import calculate_risk_score


DETECT_ANGLE_01 = {
    "left_elbow": 160, "right_elbow": 160,
    "left_shoulder": 40, "right_shoulder": 40,
    "left_knee": 90, "right_knee": 90,
    "left_hip": 100, "right_hip": 100
}

DETECT_ANGLE_02 = {
    "left_elbow": 169.1, "right_elbow": 162.7,
    "left_shoulder": 0.9, "right_shoulder": 2.1,
    "left_knee": 173.4, "right_knee": 162.2,
    "left_hip": 163.8, "right_hip": 173.9
}

STANDARD_ANGLE = {
    "left_elbow": 160, "right_elbow": 160,
    "left_shoulder": 40, "right_shoulder": 40,
    "left_knee": 90, "right_knee": 90,
    "left_hip": 100, "right_hip": 100
}


class TestCalculateRiskScore:

    def test_exact_match_returns_0(self):
        result = calculate_risk_score(DETECT_ANGLE_01, STANDARD_ANGLE)

        assert result == 0.0

    def test_deviation_returns_correct_risk_score(self):
        result = calculate_risk_score(DETECT_ANGLE_02, STANDARD_ANGLE)

        assert abs(result - 47.7625) < 1e-4