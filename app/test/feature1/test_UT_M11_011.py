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


def _mock_connection(
    reference_data=MOCK_REFERENCE,
    frames_data=MOCK_FRAMES,
    metrics_data=MOCK_METRICS
):
    mock_conn = MagicMock()
    calls = {"n": 0}

    def execute_side_effect(stmt, params=None):
        n = calls["n"]
        calls["n"] += 1

        result = MagicMock()
        if n == 0:
            result.mappings.return_value.all.return_value = reference_data
        elif n == 1:
            result.mappings.return_value.all.return_value = frames_data
        else:
            result.mappings.return_value.all.return_value = metrics_data
        return result

    mock_conn.execute.side_effect = execute_side_effect
    return mock_conn


def _patch_get_connection(mock_conn):
    mock_get_connection = MagicMock()
    mock_get_connection.return_value.__enter__.return_value = mock_conn
    mock_get_connection.return_value.__exit__.return_value = False

    return patch(
        "app.config.db.get_connection",
        mock_get_connection,
    )


class TestGetReferenceAnglesFromDb:

    def test_returns_exercise_name_average_angles_and_angle_sequence(
        self
    ):
        mock_conn = _mock_connection()

        with _patch_get_connection(mock_conn):
            result = get_reference_angles_from_db(1)

        assert result["exercise_name"] == "Squat_men"
        assert isinstance(result["average_angles"], dict)
        assert isinstance(result["angle_sequence"], list)
        assert len(result["angle_sequence"]) == 2

    def test_average_angles_correctly_calculated(
        self
    ):
        mock_conn = _mock_connection()

        with _patch_get_connection(mock_conn):
            result = get_reference_angles_from_db(1)

        expected_left_hip = (163.9 + 164.3) / 2
        assert abs(result["average_angles"]["left_hip"] - expected_left_hip) < 1e-4

    def test_raises_value_error_when_no_frames_found(
        self
    ):
        mock_conn = _mock_connection(frames_data=[])

        with _patch_get_connection(mock_conn):
            with pytest.raises(ValueError) as exc_info:
                get_reference_angles_from_db(999)

        assert "No frames found for reference_video_id=999" in str(exc_info.value)

    def test_raises_value_error_when_no_metrics_found(
        self
    ):
        mock_conn = _mock_connection(metrics_data=[])

        with _patch_get_connection(mock_conn):
            with pytest.raises(ValueError) as exc_info:
                get_reference_angles_from_db(1)

        assert "No metrics found for frame_ids=" in str(exc_info.value)

