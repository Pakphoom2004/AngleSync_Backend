from typing import Any, Dict, List, Optional

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
            .select("session_id, session_name, reference_video_id, accuracy_score, analysis_date")
            .eq("user_id", user_id)
        )

        if search_term:
            query = query.ilike("session_name", f"%{search_term}%")

        ascending = sort_order == "asc"
        query = query.order("analysis_date", desc=not ascending)

        response = query.execute()
    except Exception:
        raise HistoryException()

    return response.data or []


def session_detail(
        supabase: Client,
        user_id: int,
        session_id: int,
) -> Dict[str, Any]:
    try:
        session_response = (
            supabase.table("analysis_sessions")
            .select("session_name, video_user_url, accuracy_score,reference_video_id")
            .eq("session_id", session_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )

        if not session_response.data:
            raise HistoryException()

        session = session_response.data[0]

        risk_frames_response = (
            supabase.table("risk_frames")
            .select(
                "frame_id, session_id, frame_number, risk_percentage, "
                "skeleton_overlay_url, joint_coordinates"
            )
            .eq("session_id", session_id)
            .order("frame_number")
            .execute()
        )

        feedback_response = (
            supabase.table("feedbacks")
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

    # 💥 ปรับปรุงส่วนจัดโครงสร้าง risk_frames ให้รองรับทุกชื่อ Key ที่ Flutter อาจใช้
    raw_risk_frames = risk_frames_response.data or []
    formatted_risk_frames = []

    for frame in raw_risk_frames:
        raw_url = frame.get("skeleton_overlay_url") or ""
        
        # กรองและทำความสะอาด URL กรณีเป็น localhost หรือ path เก่า
        clean_url = raw_url
        if "127.0.0.1" in raw_url or "localhost" in raw_url or raw_url.startswith("outputs/"):
            clean_url = ""

        formatted_risk_frames.append({
            "frame_id": frame.get("frame_id"),
            "session_id": frame.get("session_id"),
            "frame_number": frame.get("frame_number"),
            "risk_percentage": frame.get("risk_percentage"),
            "skeleton_overlay_url": clean_url,
            "highest_risk_image_url": clean_url,  # 💥 เพิ่ม Key นี้ให้ตรงกับ UI
            "image": clean_url,                   # 💥 เพิ่ม Key สำรอง
            "image_url": clean_url,              # 💥 เพิ่ม Key สำรอง
            "joint_coordinates": frame.get("joint_coordinates")
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


def delete_session(
        supabase: Client,
        user_id: int,
        session_id: int,
) -> Dict[str, bool]:
    try:
        session_response = (
            supabase.table("analysis_sessions")
            .select("session_id")
            .eq("session_id", session_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )

        if not session_response.data:
            raise SessionDeleteFailedException()

        supabase.table("risk_frames").delete().eq("session_id", session_id).execute()
        supabase.table("feedbacks").delete().eq("session_id", session_id).execute()

        delete_response = (
            supabase.table("analysis_sessions")
            .delete()
            .eq("session_id", session_id)  
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