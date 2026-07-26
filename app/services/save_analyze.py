from datetime import datetime, timezone
from typing import Any, Dict, List

from supabase import Client

from app.exceptions.save_transaction_failed_exception import SaveTransactionFailedException


def save_analysis_result(
    supabase: Client,
    user_id: int,
    session_name: str,
    reference_video_id: int,
    video_user_url: str,
    accuracy_score: float,
    risk_frames: List[Dict[str, Any]],
    feedback: Dict[str, Any],
) -> Dict[str, Any]:

    session_id = None

    try:
        saved_at = datetime.now(timezone.utc).isoformat()

        session_response = (
            supabase.table("analysis_sessions")
            .insert(
                {
                    "user_id": user_id,
                    "session_name": session_name,
                    "reference_video_id": reference_video_id,
                    "video_user_url": video_user_url,
                    "accuracy_score": accuracy_score,
                    "saved_at": saved_at,
                }
            )
            .execute()
        )

        if not session_response.data:
            raise SaveTransactionFailedException()

        session_id = session_response.data[0]["id"]

        if risk_frames:
            risk_frame_rows = [
                {
                    "session_id": session_id,
                    "frame_number": frame.get("frame_number"),
                    "risk_percentage": frame.get("risk_percentage"),
                    "skeleton_overlay_url": frame.get("skeleton_overlay_url"),
                    "joint_coordinates": frame.get("joint_coordinates"),
                }
                for frame in risk_frames
            ]
            risk_frames_response = (
                supabase.table("risk_frames").insert(risk_frame_rows).execute()
            )

            if not risk_frames_response.data:
                raise SaveTransactionFailedException()

        feedback_response = (
            supabase.table("feedback")
            .insert(
                {
                    "session_id": session_id,
                    "form_summary": feedback.get("form_summary"),
                    "injury_risk": feedback.get("injury_risk"),
                    "corrective_cues": feedback.get("corrective_cues"),
                    "practice_plan": feedback.get("practice_plan"),
                }
            )
            .execute()
        )

        if not feedback_response.data:
            raise SaveTransactionFailedException()

        return {
            "success": True,
            "session_id": session_id,
            "session_name": session_name,
            "saved_at": saved_at,
        }

    except SaveTransactionFailedException:
        _cleanup_partial_save(supabase, session_id)
        raise
    except Exception:
        _cleanup_partial_save(supabase, session_id)
        raise SaveTransactionFailedException()


def _cleanup_partial_save(supabase: Client, session_id: int) -> None:

    if session_id is None:
        return

    try:
        supabase.table("risk_frames").delete().eq("session_id", session_id).execute()
        supabase.table("feedback").delete().eq("session_id", session_id).execute()
        supabase.table("analysis_sessions").delete().eq("id", session_id).execute()
    except Exception:

        pass