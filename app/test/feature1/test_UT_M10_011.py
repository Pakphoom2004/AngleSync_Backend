import pytest
from unittest.mock import MagicMock

from app.services.pose_detection import _update_progress


class TestReportProgress:

    def test_callback_called_with_correct_progress_data(self):
        mock_callback = MagicMock()

        _update_progress(mock_callback, 50, "Processing frame 1")

        mock_callback.assert_called_once_with({
            "step":    "detecting",
            "message": "Processing frame 1",
            "percent": 50
        })

    def test_none_callback_returns_none(self):
        result = _update_progress(None, 50, "Processing frame 1")

        assert result is None