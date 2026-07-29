from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from supabase import Client

from app.exceptions.save_transaction_failed_exception import SaveTransactionFailedException
from app.services.frame_cache import get_keypoints

logger = logging.getLogger(__name__)


def _first_present(data: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return None


def _text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(item) for item in value if item is not None)
    if isinstance(value, dict):
        return str(value)
    return str(value)


def _resolve_joint_coordinates(frame: Dict[str, Any], skeleton_overlay_url: Any) -> Any:
    direct_value = frame.get("joint_coordinates") or frame.get("keypoints")
    if direct_value:
        return direct_value

    if skeleton_overlay_url:
        filename = os.path.basename(str(skeleton_overlay_url))
        cached_keypoints = get_keypoints(filename)
        if cached_keypoints:
            return cached_keypoints

    return {}


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
        session_response = (
            supabase.table("analysis_sessions")
            .insert(
                {
                    "user_id": user_id,
                    "session_name": session_name,
                    "reference_video_id": reference_video_id,
                    "video_user_url": video_user_url,
                    "accuracy_score": accuracy_score,
                }
            )
            .execute()
        )

        if not session_response.data:
            raise SaveTransactionFailedException()

        session_id = session_response.data[0]["session_id"]

        if risk_frames:
            risk_frame_rows = []
            for frame in risk_frames:
                skeleton_overlay_url = _first_present(
                    frame,
                    "skeleton_overlay_url",
                    "image_url"
                )
                risk_frame_rows.append(
                    {
                        "session_id": session_id,
                        "frame_number": _first_present(frame, "frame_number", "frame"),
                        "risk_percentage": _first_present(
                            frame, "risk_percentage", "risk", "risk_score"
                        ),
                        "skeleton_overlay_url": skeleton_overlay_url,
                        "joint_coordinates": _resolve_joint_coordinates(
                            frame, skeleton_overlay_url
                        ),
                    }
                )

            risk_frames_response = (
                supabase.table("risk_frames").insert(risk_frame_rows).execute()
            )

            if not risk_frames_response.data:
                raise SaveTransactionFailedException()

        feedback_response = (
            supabase.table("feedbacks")
            .insert(
                {
                    "session_id": session_id,
                    "form_summary": _text_value(feedback.get("form_summary")),
                    "injury_risk": _text_value(feedback.get("injury_risk")),
                    "corrective_cues": _text_value(feedback.get("corrective_cues")),
                    "practice_plan": _text_value(feedback.get("practice_plan")),
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
        }

    except SaveTransactionFailedException:
        _cleanup_partial_save(supabase, session_id)
        raise
    except Exception:
        logger.exception("Failed to save analysis result")
        _cleanup_partial_save(supabase, session_id)
        raise SaveTransactionFailedException()


def _cleanup_partial_save(supabase: Client, session_id: int) -> None:

    if session_id is None:
        return

    try:
        supabase.table("risk_frames").delete().eq("session_id", session_id).execute()
        supabase.table("feedbacks").delete().eq("session_id", session_id).execute()
        supabase.table("analysis_sessions").delete().eq("session_id", session_id).execute()
    except Exception:
        pass