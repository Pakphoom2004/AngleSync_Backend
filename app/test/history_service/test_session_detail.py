import pytest
from unittest.mock import MagicMock, patch
from app.services.history_user.history_service import session_detail
from app.exceptions.history_exception import HistoryException


RISK_FRAME = {
    "frame_id": 1,
    "session_id": 101,
    "frame_number": 45,
    "risk_percentage": 60.0,
    "skeleton_overlay_url": "https://storage.example.com/overlays/frame45.png",
    "joint_coordinates": {"left_knee": [120, 340]},
}

FEEDBACK = {
    "form_summary": "Good squat form overall.",
    "injury_risk": "Low",
    "corrective_cues": "Keep knees aligned.",
    "practice_plan": "Practice 3x weekly.",
}


def _mock_connection(
    session_row=None,
    session_raises=None,
    risk_frames_data=None,
    feedback_row=None,
):
    mock_conn = MagicMock()
    calls = {"n": 0}

    def execute_side_effect(*args, **kwargs):
        n = calls["n"]
        calls["n"] += 1

        if n == 0:
            if session_raises:
                raise session_raises
            result = MagicMock()
            result.mappings.return_value.first.return_value = session_row
            return result

        if n == 1:
            result = MagicMock()
            result.mappings.return_value.all.return_value = risk_frames_data or []
            return result

        result = MagicMock()
        result.mappings.return_value.first.return_value = feedback_row
        return result

    mock_conn.execute.side_effect = execute_side_effect
    return mock_conn


def _patch_get_connection(mock_conn):
    mock_get_connection = MagicMock()
    mock_get_connection.return_value.__enter__.return_value = mock_conn
    mock_get_connection.return_value.__exit__.return_value = False
    return patch(
        "app.services.history_user.history_service.get_connection",
        mock_get_connection,
    )


# UT-01
def test_session_detail_returns_full_detail_when_video_available():
    session_row = {
        "session_name": "Squat Session 1",
        "video_user_url": "https://storage.example.com/videos/squat.mp4",
        "accuracy_score": 87.5,
        "reference_video_id": 10,
    }
    mock_conn = _mock_connection(
        session_row=session_row,
        risk_frames_data=[RISK_FRAME],
        feedback_row=FEEDBACK,
    )

    with _patch_get_connection(mock_conn):
        result = session_detail(user_id=1, session_id=101)

    assert result["session_id"] == 101
    assert result["session_name"] == "Squat Session 1"
    assert result["video_user_url"] == "https://storage.example.com/videos/squat.mp4"
    assert result["reference_video_id"] == 10
    assert result["analysis_result"]["accuracy_score"] == 87.5
    assert len(result["analysis_result"]["risk_frames"]) == 1
    assert result["analysis_result"]["feedback"] == FEEDBACK


# UT-02
def test_session_detail_returns_placeholder_when_video_url_is_none():
    session_row = {
        "session_name": "Squat Session 2",
        "video_user_url": None,
        "accuracy_score": 72.0,
        "reference_video_id": 10,
    }
    mock_conn = _mock_connection(
        session_row=session_row,
        risk_frames_data=[RISK_FRAME],
        feedback_row=FEEDBACK,
    )

    with _patch_get_connection(mock_conn):
        result = session_detail(user_id=1, session_id=102)

    assert result["session_name"] == "Squat Session 2"
    assert result["video_user_url"] == "Video not available."
    assert result["reference_video_id"] == 10
    assert result["analysis_result"]["accuracy_score"] == 72.0
    assert len(result["analysis_result"]["risk_frames"]) == 1
    assert result["analysis_result"]["feedback"] == FEEDBACK


# UT-03
def test_session_detail_raises_when_db_query_fails():
    mock_conn = _mock_connection(
        session_raises=Exception("DB connection error"),
    )

    with _patch_get_connection(mock_conn):
        with pytest.raises(HistoryException) as exc_info:
            session_detail(user_id=1, session_id=101)

    assert str(exc_info.value) == "Couldn't load your history right now. Please try again."