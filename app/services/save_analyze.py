from __future__ import annotations

import logging
import os
from typing import Any, Dict, List


from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import JSONB

from app.config.db import get_connection
from app.exceptions.save_transaction_failed_exception import SaveTransactionFailedException
from app.services.frame_cache import get_keypoints

logger = logging.getLogger(__name__)


def _first_present(data: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = data.get(key)
        if value is not None and str(value).strip() != "":
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
        with get_connection() as conn:
            session_result = conn.execute(
                text(
                    """
                    INSERT INTO analysis_sessions
                        (user_id, session_name, reference_video_id, video_user_url, accuracy_score)
                    VALUES
                        (:user_id, :session_name, :reference_video_id, :video_user_url, :accuracy_score)
                    RETURNING session_id
                    """
                ),
                {
                    "user_id": user_id,
                    "session_name": session_name,
                    "reference_video_id": reference_video_id,
                    "video_user_url": video_user_url,
                    "accuracy_score": accuracy_score,
                },
            )

            session_row = session_result.mappings().first()
            if not session_row:
                raise SaveTransactionFailedException()

            session_id = session_row["session_id"]

            if risk_frames:
                for frame in risk_frames:
                    skeleton_overlay_url = _first_present(
                        frame,
                        "highest_risk_image_url",
                        "image",
                        "skeleton_overlay_url",
                        "image_url"
                    )

                    if skeleton_overlay_url and ("127.0.0.1" in str(skeleton_overlay_url) or "localhost" in str(skeleton_overlay_url)):
                        skeleton_overlay_url = ""

                    risk_frame_result = conn.execute(
                        text(
                            """
                            INSERT INTO risk_frames
                                (session_id, frame_number, risk_percentage, skeleton_overlay_url, joint_coordinates)
                            VALUES
                                (:session_id, :frame_number, :risk_percentage, :skeleton_overlay_url, :joint_coordinates)
                            RETURNING frame_id
                            """
                        ).bindparams(bindparam("joint_coordinates", type_=JSONB)),
                        {
                            "session_id": session_id,
                            "frame_number": _first_present(frame, "frame_number", "frame"),
                            "risk_percentage": _first_present(
                                frame, "risk_percentage", "risk", "risk_score"
                            ),
                            "skeleton_overlay_url": skeleton_overlay_url or "",
                            "joint_coordinates": _resolve_joint_coordinates(
                                frame, skeleton_overlay_url
                            ),
                        },
                    )

                    if not risk_frame_result.mappings().first():
                        raise SaveTransactionFailedException()

            feedback_result = conn.execute(
                text(
                    """
                    INSERT INTO feedbacks
                        (session_id, form_summary, injury_risk, corrective_cues, practice_plan)
                    VALUES
                        (:session_id, :form_summary, :injury_risk, :corrective_cues, :practice_plan)
                    RETURNING feedback_id
                    """
                ),
                {
                    "session_id": session_id,
                    "form_summary": _text_value(feedback.get("form_summary")),
                    "injury_risk": _text_value(feedback.get("injury_risk")),
                    "corrective_cues": _text_value(feedback.get("corrective_cues")),
                    "practice_plan": _text_value(feedback.get("practice_plan")),
                },
            )

            if not feedback_result.mappings().first():
                raise SaveTransactionFailedException()

        return {
            "success": True,
            "session_id": session_id,
            "session_name": session_name,
        }

    except SaveTransactionFailedException:
        _cleanup_partial_save(session_id)
        raise
    except Exception:
        logger.exception("Failed to save analysis result")
        _cleanup_partial_save(session_id)
        raise SaveTransactionFailedException()


def _cleanup_partial_save(session_id: int) -> None:

    if session_id is None:
        return

    try:
        with get_connection() as conn:
            conn.execute(
                text("DELETE FROM risk_frames WHERE session_id = :session_id"),
                {"session_id": session_id},
            )
            conn.execute(
                text("DELETE FROM feedbacks WHERE session_id = :session_id"),
                {"session_id": session_id},
            )
            conn.execute(
                text("DELETE FROM analysis_sessions WHERE session_id = :session_id"),
                {"session_id": session_id},
            )
    except Exception:
        pass