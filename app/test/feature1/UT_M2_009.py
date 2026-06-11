import pytest
from unittest.mock import MagicMock, patch

from app.services.pose_detection import verify_video_duration


def _make_cap(fps: float, total_frames: float):
    cap = MagicMock()
    cap.get.side_effect = lambda prop: fps if prop == 5 else total_frames
    return cap


class TestVerifyVideoDuration:

    @patch("app.services.analysis_results.cv2.VideoCapture")
    def test_duration_within_limit_returns_none(self, mock_cap_cls):
        mock_cap_cls.return_value = _make_cap(fps=30.0, total_frames=900)

        result = verify_video_duration("exercise_vid.mp4")

        assert result is None

    @patch("app.services.analysis_results.cv2.VideoCapture")
    def test_duration_exactly_at_limit_returns_none(self, mock_cap_cls):
        mock_cap_cls.return_value = _make_cap(fps=30.0, total_frames=1800)

        result = verify_video_duration("exercise_vid.mp4")

        assert result is None

    @patch("app.services.analysis_results.cv2.VideoCapture")
    def test_duration_exceeds_limit_returns_error_message(self, mock_cap_cls):
        mock_cap_cls.return_value = _make_cap(fps=30.0, total_frames=2700)

        result = verify_video_duration("exercise_vid.mp4")

        assert result == "Video exceeds 60 seconds."