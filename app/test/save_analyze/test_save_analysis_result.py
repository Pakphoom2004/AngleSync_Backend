import pytest
from unittest.mock import MagicMock, patch
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


def _mock_connection(
    session_result=None,
    session_raises=None,
    risk_frame_results=None,
    risk_frame_raises=None,
    feedback_result=None,
    feedback_raises=None,
):
    """
    Build a connection mock whose execute() dispatches based on the SQL text,
    mirroring the (session -> risk_frames -> feedback) insert order.
    """
    mock_conn = MagicMock()
    risk_frame_iter = iter(risk_frame_results or [])

    def execute_side_effect(stmt, params=None):
        sql = str(stmt)

        if "INSERT INTO analysis_sessions" in sql:
            if session_raises:
                raise session_raises
            result = MagicMock()
            result.mappings.return_value.first.return_value = session_result
            return result

        if "INSERT INTO risk_frames" in sql:
            if risk_frame_raises:
                raise risk_frame_raises
            result = MagicMock()
            result.mappings.return_value.first.return_value = next(risk_frame_iter, None)
            return result

        if "INSERT INTO feedbacks" in sql:
            if feedback_raises:
                raise feedback_raises
            result = MagicMock()
            result.mappings.return_value.first.return_value = feedback_result
            return result

        # cleanup DELETE queries
        return MagicMock()

    mock_conn.execute.side_effect = execute_side_effect
    return mock_conn


def _patch_get_connection(mock_conn):
    mock_get_connection = MagicMock()
    mock_get_connection.return_value.__enter__.return_value = mock_conn
    mock_get_connection.return_value.__exit__.return_value = False
    return patch(
        "app.services.save_analyze.get_connection",
        mock_get_connection,
    )


# UT-01
def test_save_analysis_result_success():
    mock_conn = _mock_connection(
        session_result={"session_id": 42},
        risk_frame_results=[{"frame_id": 1}],
        feedback_result={"feedback_id": 1},
    )

    with _patch_get_connection(mock_conn):
        result = save_analysis_result(
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
    mock_conn = _mock_connection(
        session_raises=Exception("DB connection error"),
    )

    with _patch_get_connection(mock_conn):
        with pytest.raises(SaveTransactionFailedException) as exc_info:
            save_analysis_result(
                user_id=USER_ID,
                session_name=SESSION_NAME,
                reference_video_id=REFERENCE_VIDEO_ID,
                video_user_url=VIDEO_USER_URL,
                accuracy_score=ACCURACY_SCORE,
                risk_frames=RISK_FRAMES,
                feedback=FEEDBACK_SIMPLE,
            )

    assert str(exc_info.value) == "Unable to save result. Please try again."


# UT-03
def test_save_analysis_result_raises_and_cleans_up_when_risk_frames_insert_fails():
    mock_conn = _mock_connection(
        session_result={"session_id": 42},
        risk_frame_raises=Exception("schema violation"),
    )

    with _patch_get_connection(mock_conn):
        with pytest.raises(SaveTransactionFailedException) as exc_info:
            save_analysis_result(
                user_id=USER_ID,
                session_name=SESSION_NAME,
                reference_video_id=REFERENCE_VIDEO_ID,
                video_user_url=VIDEO_USER_URL,
                accuracy_score=ACCURACY_SCORE,
                risk_frames=RISK_FRAMES,
                feedback=FEEDBACK_SIMPLE,
            )

    assert str(exc_info.value) == "Unable to save result. Please try again."


# UT-04
def test_save_analysis_result_raises_when_foreign_key_invalid():
    mock_conn = _mock_connection(
        session_raises=Exception("FK constraint violation: user_id not found"),
    )

    with _patch_get_connection(mock_conn):
        with pytest.raises(SaveTransactionFailedException) as exc_info:
            save_analysis_result(
                user_id=99999,
                session_name=SESSION_NAME,
                reference_video_id=REFERENCE_VIDEO_ID,
                video_user_url=VIDEO_USER_URL,
                accuracy_score=ACCURACY_SCORE,
                risk_frames=RISK_FRAMES,
                feedback=FEEDBACK_SIMPLE,
            )

    assert str(exc_info.value) == "Unable to save result. Please try again."
