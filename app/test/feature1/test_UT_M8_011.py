import math
import pytest

from app.services.motion_analysis import calculate_angle


class TestCalculateAngle:

    def test_right_angle_returns_90(self):
        start_point  = (0.0, 1.0)
        middle_point = (0.0, 0.0)
        end_point    = (1.0, 0.0)

        result = calculate_angle(start_point, middle_point, end_point)

        assert math.isclose(result, 90.0, abs_tol=1e-4)

    def test_straight_line_returns_180(self):
        start_point  = (0.0, 0.0)
        middle_point = (1.0, 0.0)
        end_point    = (2.0, 0.0)

        result = calculate_angle(start_point, middle_point, end_point)

        assert math.isclose(result, 180.0, abs_tol=1e-4)

    def test_zero_length_vector_returns_0(self):
        start_point  = (0.0, 0.0)
        middle_point = (0.0, 0.0)
        end_point    = (1.0, 0.0)

        result = calculate_angle(start_point, middle_point, end_point)

        assert result == 0.0