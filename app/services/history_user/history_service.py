from datetime import date
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from app.config.db import get_connection
from app.exceptions.history_exception import HistoryException
from app.exceptions.session_delete_failed_exception import SessionDeleteFailedException

def history_list(
    user_id: int,
    search_term: Optional[str] = None,
    sort_order: str = "desc",
) -> List[Dict[str, Any]]:
    ascending = sort_order.lower() == "asc"
    order_clause = "ASC" if ascending else "DESC"

    query = (
        "SELECT session_id, session_name, reference_video_id, accuracy_score, analysis_date "
        "FROM analysis_sessions WHERE user_id = :user_id"
    )
    params: Dict[str, Any] = {"user_id": user_id}

    if search_term and search_term.strip():
        query += " AND session_name ILIKE :search_term"
        params["search_term"] = f"%{search_term.strip()}%"

    query += f" ORDER BY analysis_date {order_clause}"

    try:
        with get_connection() as conn:
            rows = conn.execute(text(query), params).mappings().all()
    except Exception:
        raise HistoryException()

    return [dict(row) for row in rows]

def session_detail(
    user_id: int,
    session_id: int,
) -> Dict[str, Any]:
    try:
        with get_connection() as conn:
            session_row = conn.execute(
                text(
                    """
                    SELECT session_name, video_user_url, accuracy_score, reference_video_id
                    FROM analysis_sessions
                    WHERE session_id = :session_id AND user_id = :user_id
                    LIMIT 1
                    """
                ),
                {"session_id": session_id, "user_id": user_id},
            ).mappings().first()

            if not session_row:
                raise HistoryException()

            session = dict(session_row)

            risk_frames_rows = conn.execute(
                text(
                    """
                    SELECT frame_id, session_id, frame_number, risk_percentage,
                           skeleton_overlay_url, joint_coordinates
                    FROM risk_frames
                    WHERE session_id = :session_id
                    ORDER BY frame_number
                    """
                ),
                {"session_id": session_id},
            ).mappings().all()

            feedback_row = conn.execute(
                text(
                    """
                    SELECT form_summary, injury_risk, corrective_cues, practice_plan
                    FROM feedbacks
                    WHERE session_id = :session_id
                    LIMIT 1
                    """
                ),
                {"session_id": session_id},
            ).mappings().first()
    except HistoryException:
        raise
    except Exception:
        raise HistoryException()

    video_user_url = session.get("video_user_url") or "Video not available."
    feedback_data = dict(feedback_row) if feedback_row else {}

    raw_risk_frames = [dict(row) for row in risk_frames_rows]
    formatted_risk_frames = []

    for frame in raw_risk_frames:
        raw_url = frame.get("skeleton_overlay_url") or ""
        clean_url = raw_url
        if "127.0.0.1" in raw_url or "localhost" in raw_url or raw_url.startswith("outputs/"):
            clean_url = ""

        formatted_risk_frames.append({
            "frame_id": frame.get("frame_id"),
            "session_id": frame.get("session_id"),
            "frame_number": frame.get("frame_number"),
            "risk_percentage": frame.get("risk_percentage"),
            "skeleton_overlay_url": clean_url,
            "highest_risk_image_url": clean_url,
            "image": clean_url,
            "image_url": clean_url,
            "joint_coordinates": frame.get("joint_coordinates"),
        })

    analysis_result = {
        "accuracy_score": session.get("accuracy_score"),
        "risk_frames": formatted_risk_frames,
        "feedback": feedback_data,
    }

    return {
        "session_id": session_id,
        "session_name": session.get("session_name"),
        "video_user_url": video_user_url,
        "reference_video_id": session.get("reference_video_id"),
        "analysis_result": analysis_result,
    }

def search_history_by_session_name(
    user_id: int,
    search_term: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return history_list(user_id=user_id, search_term=search_term, sort_order="desc")


def filter_and_sort_history_by_date(
    user_id: int,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    sort_order: str = "desc"
) -> List[Dict[str, Any]]:
    ascending = sort_order.lower() == "asc"
    order_clause = "ASC" if ascending else "DESC"

    query = (
        "SELECT session_id, session_name, reference_video_id, accuracy_score, analysis_date "
        "FROM analysis_sessions WHERE user_id = :user_id"
    )
    params: Dict[str, Any] = {"user_id": user_id}

    if start_date:
        query += " AND DATE(analysis_date) >= :start_date"
        params["start_date"] = start_date

    if end_date:
        query += " AND DATE(analysis_date) <= :end_date"
        params["end_date"] = end_date

    query += f" ORDER BY analysis_date {order_clause}"

    try:
        with get_connection() as conn:
            rows = conn.execute(text(query), params).mappings().all()
    except Exception:
        raise HistoryException()

    return [dict(row) for row in rows]

def delete_session(
    user_id: int,
    session_id: int,
) -> Dict[str, bool]:
    try:
        with get_connection() as conn:
            session_row = conn.execute(
                text(
                    """
                    SELECT session_id
                    FROM analysis_sessions
                    WHERE session_id = :session_id AND user_id = :user_id
                    LIMIT 1
                    """
                ),
                {"session_id": session_id, "user_id": user_id},
            ).mappings().first()

            if not session_row:
                raise SessionDeleteFailedException()

            conn.execute(
                text("DELETE FROM risk_frames WHERE session_id = :session_id"),
                {"session_id": session_id},
            )
            conn.execute(
                text("DELETE FROM feedbacks WHERE session_id = :session_id"),
                {"session_id": session_id},
            )

            delete_result = conn.execute(
                text(
                    """
                    DELETE FROM analysis_sessions
                    WHERE session_id = :session_id AND user_id = :user_id
                    RETURNING session_id
                    """
                ),
                {"session_id": session_id, "user_id": user_id},
            )
            deleted = delete_result.mappings().all()

            if not deleted:
                raise SessionDeleteFailedException()
    except SessionDeleteFailedException:
        raise
    except Exception:
        raise SessionDeleteFailedException()

    return {"success": True}