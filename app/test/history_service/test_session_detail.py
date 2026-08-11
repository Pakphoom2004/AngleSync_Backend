import pytest
from unittest.mock import MagicMock
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


def _make_supabase_mock(
    session_data=None,
    session_raises=None,
    risk_frames_data=None,
    feedback_data=None,
):
    mock_supabase = MagicMock()
    call_count = {"n": 0}

    def table_side_effect(table_name):
        builder = MagicMock()
        builder.select.return_value = builder
        builder.eq.return_value = builder
        builder.limit.return_value = builder
        builder.order.return_value = builder

        n = call_count["n"]
        call_count["n"] += 1

        if table_name == "analysis_sessions" and n == 0:
            if session_raises:
                builder.execute.side_effect = session_raises
            else:
                result = MagicMock()
                result.data = session_data
                builder.execute.return_value = result

        elif table_name == "risk_frames":
            result = MagicMock()
            result.data = risk_frames_data or []
            builder.execute.return_value = result

        elif table_name == "feedbacks":
            result = MagicMock()
            result.data = feedback_data or []
            builder.execute.return_value = result

        return builder

    mock_supabase.table.side_effect = table_side_effect
    return mock_supabase


# UT-01
def test_session_detail_returns_full_detail_when_video_available():
    session_data = [{
        "session_name": "Squat Session 1",
        "video_user_url": "https://storage.example.com/videos/squat.mp4",
        "accuracy_score": 87.5,
        "reference_video_id": 10,
    }]
    mock_supabase = _make_supabase_mock(
        session_data=session_data,
        risk_frames_data=[RISK_FRAME],
        feedback_data=[FEEDBACK],
    )

    result = session_detail(mock_supabase, user_id=1, session_id=101)

    assert result["session_id"] == 101
    assert result["session_name"] == "Squat Session 1"
    assert result["video_user_url"] == "https://storage.example.com/videos/squat.mp4"
    assert result["reference_video_id"] == 10
    assert result["analysis_result"]["accuracy_score"] == 87.5
    assert len(result["analysis_result"]["risk_frames"]) == 1
    assert result["analysis_result"]["feedback"] == FEEDBACK


# UT-02
def test_session_detail_returns_placeholder_when_video_url_is_none():
    session_data = [{
        "session_name": "Squat Session 2",
        "video_user_url": None,
        "accuracy_score": 72.0,
        "reference_video_id": 10,
    }]
    mock_supabase = _make_supabase_mock(
        session_data=session_data,
        risk_frames_data=[RISK_FRAME],
        feedback_data=[FEEDBACK],
    )

    result = session_detail(mock_supabase, user_id=1, session_id=102)

    assert result["session_name"] == "Squat Session 2"
    assert result["video_user_url"] == "Video not available."
    assert result["reference_video_id"] == 10
    assert result["analysis_result"]["accuracy_score"] == 72.0
    assert len(result["analysis_result"]["risk_frames"]) == 1
    assert result["analysis_result"]["feedback"] == FEEDBACK


# UT-03
def test_session_detail_raises_when_db_query_fails():
    mock_supabase = _make_supabase_mock(
        session_raises=Exception("DB connection error"),
    )

    with pytest.raises(HistoryException) as exc_info:
        session_detail(mock_supabase, user_id=1, session_id=101)

    assert str(exc_info.value) == "Couldn't load your history right now. Please try again."