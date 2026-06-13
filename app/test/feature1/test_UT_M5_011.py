import pytest
from unittest.mock import patch, MagicMock

from app.repository.reference_repository import get_standard_angle_from_db


MOCK_AVERAGE_ANGLES = {
    "left_hip": 164.1, "left_knee": 173.55,
    "right_hip": 173.65, "left_ankle": 165.55,
    "left_elbow": 169.2, "left_wrist": 64.85,
    "right_knee": 162.3, "right_ankle": 171.7,
    "right_elbow": 163.05, "right_wrist": 20.45,
    "left_shoulder": 0.7, "right_shoulder": 1.85
}


class TestGetStandardAngleFromDb:

    @patch("app.repository.reference_repository.get_reference_angles_from_db")
    def test_returns_average_angles_dict(self, mock_get_reference):
        mock_get_reference.return_value = {
            "exercise_name": "Squat_men",
            "average_angles": MOCK_AVERAGE_ANGLES,
            "angle_sequence": []
        }

        result = get_standard_angle_from_db(1)

        assert isinstance(result, dict)
        assert result == MOCK_AVERAGE_ANGLES

    @patch("app.repository.reference_repository.get_reference_angles_from_db")
    def test_raises_value_error_when_no_frames_found(self, mock_get_reference):
        mock_get_reference.side_effect = ValueError(
            "No frames found for reference_video_id=999"
        )

        with pytest.raises(ValueError) as exc_info:
            get_standard_angle_from_db(999)

        assert "No frames found for reference_video_id=999" in str(exc_info.value)