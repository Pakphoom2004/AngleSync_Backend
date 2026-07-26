from typing import Any, Dict, List, Optional

from requests import session
from supabase import Client

from app.exceptions.history_exception import HistoryException
from app.exceptions.session_delete_failed_exception import SessionDeleteFailedException

def history_list(
    supabase: Client,
    user_id: int,
    search_term: Optional[str],
    sort_order: str,
) -> List[Dict[str, Any]]:
    try:
        query = (
            supabase.table("analysis_sessions")
            .select("session_name, saved_at")
            .eq("user_id", user_id)
        )

        if search_term:
            query = query.ilike("session_name", f"%{search_term}%")

        ascending = sort_order == "asc"
        query = query.order("saved_at", desc=not ascending)

        response = query.execute()
    except Exception:
        raise HistoryException()

    records = response.data or []

    if not records:
        if search_term:
            raise HistoryException("No results found.")
        raise HistoryException(
            "No history yet. Start a Smart Scan to see your results here."
        )

    return records

# session_detail
def session_detail(
        supabase: Client,
        user_id: int,
        session_id: int,
) -> Dict[str, Any]:
    try:
        session_response = (
            supabase.table("analysis_sessions")
            .select("session_name, video_user_url, accuracy_score")
            .eq("id", session_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )

        if not session_response.data:
            raise HistoryException()

        session = session_response.data[0]

        risk_frames_response = (
            supabase.table("risk_frames")
            .select("frame_number, risk_percentage, skeleton_overlay_url, joint_coordinates")
            .eq("session_id", session_id)
            .execute()
        )

        feedback_response = (
            supabase.table("feedback")
            .select("form_summary, injury_risk, corrective_cues, practice_plan")
            .eq("session_id", session_id)
            .limit(1)
            .execute()
        )
    except HistoryException:
        raise
    except Exception:
        raise HistoryException()

    video_user_url = session.get("video_user_url") or "Video not available."

    feedback_data = feedback_response.data[0] if feedback_response.data else {}

    analysis_result = {
        "accuracy_score": session.get("accuracy_score"),
        "risk_frames": risk_frames_response.data or [],
        "feedback": feedback_data,
    }

    return {
        "session_name": session.get("session_name"),
        "video_user_url": video_user_url,
        "analysis_result": analysis_result,
    }


def delete_session(
        supabase: Client,
        user_id: int,
        session_id: int,
) -> Dict[str, bool]:
    try:
        session_response = (
            supabase.table("analysis_sessions")
            .select("id")
            .eq("id", session_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )

        if not session_response.data:
            raise SessionDeleteFailedException()

        supabase.table("risk_frames").delete().eq("session_id", session_id).execute()
        supabase.table("feedback").delete().eq("session_id", session_id).execute()

        delete_response = (
            supabase.table("analysis_sessions")
            .delete()
            .eq("id", session_id)
            .eq("user_id", user_id)
            .execute()
        )

        if not delete_response.data:
            raise SessionDeleteFailedException()
    except SessionDeleteFailedException:
        raise
    except Exception:
        raise SessionDeleteFailedException()

    return {"success": True}