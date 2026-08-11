import pytest
from unittest.mock import MagicMock, patch, call
from app.services.save_analyze import save_analysis_result
from app.exceptions.save_transaction_failed_exception import SaveTransactionFailedException

USER_ID = 1
SESSION_NAME = "Squat Session 1"
REFERENCE_VIDEO_ID = 10
VIDEO_USER_URL = "https://storage.example.com/videos/user1_squat.mp4"
ACCURACY_SCORE = 87.5
RISK_FRAMES = [
    {
        "frame_number": 45,
        "risk_percentage": 60.0,
        "skeleton_overlay_url": "https://storage.example.com/overlays/frame45.png",
    }
]
FEEDBACK = {
    "form_summary": "Overall good squat form with minor knee alignment issues.",
    "injury_risk": "Moderate",
    "corrective_cues": ["Keep knees aligned with toes", "Engage core throughout movement"],
    "practice_plan": "Practice bodyweight squats focusing on knee tracking, 3 sets of 12 reps daily.",
}
FEEDBACK_SIMPLE = {
    "form_summary": "Good form",
    "injury_risk": "Low",
    "corrective_cues": ["Keep back straight"],
    "practice_plan": "Practice 3x weekly",
}


def _make_supabase_mock(
    session_data=None,
    session_raises=None,
    risk_frames_data=None,
    risk_frames_raises=None,
    feedback_data=None,
    feedback_raises=None,
):
    """
    Build a Supabase client mock that wires up the builder chain
    (.table().insert().execute()) for each of the 3 tables in order.
    """
    mock_supabase = MagicMock()

    def make_execute_mock(data=None, raises=None):
        execute_mock = MagicMock()
        if raises:
            execute_mock.side_effect = raises
        else:
            result = MagicMock()
            result.data = data
            execute_mock.return_value = result
        return execute_mock

    session_execute = make_execute_mock(
        data=session_data,
        raises=session_raises,
    )
    risk_execute = make_execute_mock(
        data=risk_frames_data,
        raises=risk_frames_raises,
    )
    feedback_execute = make_execute_mock(
        data=feedback_data,
        raises=feedback_raises,
    )

    call_count = {"n": 0}

    def table_side_effect(table_name):
        builder = MagicMock()

        if table_name == "analysis_sessions" and call_count["n"] == 0:
            builder.insert.return_value.execute = session_execute
        elif table_name == "risk_frames":
            builder.insert.return_value.execute = risk_execute
        elif table_name == "feedbacks":
            builder.insert.return_value.execute = feedback_execute
        elif table_name == "analysis_sessions":
            # cleanup delete call
            builder.delete.return_value.eq.return_value.execute = MagicMock()

        call_count["n"] += 1
        return builder

    mock_supabase.table.side_effect = table_side_effect
    return mock_supabase


# UT-01
def test_save_analysis_result_success():
    mock_supabase = _make_supabase_mock(
        session_data=[{"session_id": 42}],
        risk_frames_data=[{"frame_id": 1}],
        feedback_data=[{"feedback_id": 1}],
    )

    result = save_analysis_result(
        supabase=mock_supabase,
        user_id=USER_ID,
        session_name=SESSION_NAME,
        reference_video_id=REFERENCE_VIDEO_ID,
        video_user_url=VIDEO_USER_URL,
        accuracy_score=ACCURACY_SCORE,
        risk_frames=RISK_FRAMES,
        feedback=FEEDBACK,
    )

    assert result["success"] is True
    assert result["session_id"] == 42
    assert result["session_name"] == SESSION_NAME


# UT-02
def test_save_analysis_result_raises_when_session_insert_fails():
    mock_supabase = _make_supabase_mock(
        session_raises=Exception("DB connection error"),
    )

    with pytest.raises(SaveTransactionFailedException) as exc_info:
        save_analysis_result(
            supabase=mock_supabase,
            user_id=USER_ID,
            session_name=SESSION_NAME,
            reference_video_id=REFERENCE_VIDEO_ID,
            video_user_url=VIDEO_USER_URL,
            accuracy_score=ACCURACY_SCORE,
            risk_frames=RISK_FRAMES,
            feedback=FEEDBACK_SIMPLE,
        )

    assert exc_info.value.message == "Unable to save result. Please try again."


# UT-03
def test_save_analysis_result_raises_and_cleans_up_when_risk_frames_insert_fails():
    mock_supabase = MagicMock()
    call_order = {"n": 0}

    def table_side_effect(table_name):
        builder = MagicMock()
        n = call_order["n"]
        call_order["n"] += 1

        if table_name == "analysis_sessions" and n == 0:
            session_result = MagicMock()
            session_result.data = [{"session_id": 42}]
            builder.insert.return_value.execute.return_value = session_result

        elif table_name == "risk_frames":
            builder.insert.return_value.execute.side_effect = Exception("schema violation")

        elif table_name == "analysis_sessions" and n > 0:
            # cleanup calls
            delete_builder = MagicMock()
            builder.delete.return_value = delete_builder

        return builder

    mock_supabase.table.side_effect = table_side_effect

    with pytest.raises(SaveTransactionFailedException) as exc_info:
        save_analysis_result(
            supabase=mock_supabase,
            user_id=USER_ID,
            session_name=SESSION_NAME,
            reference_video_id=REFERENCE_VIDEO_ID,
            video_user_url=VIDEO_USER_URL,
            accuracy_score=ACCURACY_SCORE,
            risk_frames=RISK_FRAMES,
            feedback=FEEDBACK_SIMPLE,
        )

    assert exc_info.value.message == "Unable to save result. Please try again."

    called_tables = [c.args[0] for c in mock_supabase.table.call_args_list]
    assert "risk_frames" in called_tables
    assert called_tables.count("analysis_sessions") >= 2


# UT-04
def test_save_analysis_result_raises_when_foreign_key_invalid():
    mock_supabase = _make_supabase_mock(
        session_raises=Exception("FK constraint violation: user_id not found"),
    )

    with pytest.raises(SaveTransactionFailedException) as exc_info:
        save_analysis_result(
            supabase=mock_supabase,
            user_id=99999,
            session_name=SESSION_NAME,
            reference_video_id=REFERENCE_VIDEO_ID,
            video_user_url=VIDEO_USER_URL,
            accuracy_score=ACCURACY_SCORE,
            risk_frames=RISK_FRAMES,
            feedback=FEEDBACK_SIMPLE,
        )

    assert str(exc_info.value) == "Unable to save result. Please try again."