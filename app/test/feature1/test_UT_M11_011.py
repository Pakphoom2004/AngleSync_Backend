import pytest
from unittest.mock import MagicMock, patch

from app.repository.reference_repository import get_reference_angles_from_db

MOCK_REFERENCE = [{"exercise_name": "Squat_men"}]

MOCK_FRAMES = [
    {"reference_frame_id": 1, "frame_sequence": 1},
    {"reference_frame_id": 2, "frame_sequence": 2},
]

MOCK_METRICS = [
    {
        "related_frame_id": 1,
        "joint_angle_data": {
            "frame": 1, "time": 0.03,
            "left_hip": 163.9, "left_knee": 173.4,
            "right_hip": 173.9, "left_ankle": 169.0,
            "left_elbow": 169.1, "left_wrist": 42.4,
            "right_knee": 162.2, "right_ankle": 173.8,
            "right_elbow": 162.7, "right_wrist": 17.2,
            "left_shoulder": 0.9, "right_shoulder": 2.1
        }
    },
    {
        "related_frame_id": 2,
        "joint_angle_data": {
            "frame": 2, "time": 0.06,
            "left_hip": 164.3, "left_knee": 173.7,
            "right_hip": 173.4, "left_ankle": 162.1,
            "left_elbow": 169.3, "left_wrist": 87.3,
            "right_knee": 162.4, "right_ankle": 169.6,
            "right_elbow": 163.4, "right_wrist": 23.7,
            "left_shoulder": 0.5, "right_shoulder": 1.6
        }
    },
]


def _make_supabase_mock(
    reference_data=MOCK_REFERENCE,
    frames_data=MOCK_FRAMES,
    metrics_data=MOCK_METRICS
):
    supabase = MagicMock()

    def table_side_effect(table_name):
        mock = MagicMock()

        if table_name == "exercise_reference":
            mock.select.return_value.eq.return_value \
                .limit.return_value.execute.return_value \
                = MagicMock(data=reference_data)

        elif table_name == "exercise_reference_frame_data":
            mock.select.return_value.eq.return_value \
                .order.return_value.execute.return_value \
                = MagicMock(data=frames_data)

        elif table_name == "exercise_reference_pose_metrics":
            mock.select.return_value.in_.return_value \
                .execute.return_value \
                = MagicMock(data=metrics_data)

        return mock

    supabase.table.side_effect = table_side_effect
    return supabase


class TestGetReferenceAnglesFromDb:

    @patch("app.config.supabase_client.supabase")
    def test_returns_exercise_name_average_angles_and_angle_sequence(
        self, mock_supabase
    ):
        mock_supabase.table.side_effect = \
            _make_supabase_mock().table.side_effect

        result = get_reference_angles_from_db(1)

        assert result["exercise_name"] == "Squat_men"
        assert isinstance(result["average_angles"], dict)
        assert isinstance(result["angle_sequence"], list)
        assert len(result["angle_sequence"]) == 2

    @patch("app.config.supabase_client.supabase")
    def test_average_angles_correctly_calculated(
        self, mock_supabase
    ):
        mock_supabase.table.side_effect = \
            _make_supabase_mock().table.side_effect

        result = get_reference_angles_from_db(1)

        expected_left_hip = (163.9 + 164.3) / 2
        assert abs(result["average_angles"]["left_hip"] - expected_left_hip) < 1e-4

    @patch("app.config.supabase_client.supabase")
    def test_raises_value_error_when_no_frames_found(
        self, mock_supabase
    ):
        mock_supabase.table.side_effect = \
            _make_supabase_mock(frames_data=[]).table.side_effect

        with pytest.raises(ValueError) as exc_info:
            get_reference_angles_from_db(999)

        assert "No frames found for reference_video_id=999" in str(exc_info.value)

    @patch("app.config.supabase_client.supabase")
    def test_raises_value_error_when_no_metrics_found(
        self, mock_supabase
    ):
        mock_supabase.table.side_effect = \
            _make_supabase_mock(metrics_data=[]).table.side_effect

        with pytest.raises(ValueError) as exc_info:
            get_reference_angles_from_db(1)

        assert "No metrics found for frame_ids=" in str(exc_info.value)