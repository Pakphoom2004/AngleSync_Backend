from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import JSONB

from app.config.db import get_connection
from app.exceptions.save_session_failed_exception import SaveSessionFailedException
from app.services.frame_cache import get_keypoints

logger = logging.getLogger(__name__)

RISK_FRAMES_BATCH_SIZE = 200

def _generate_default_session_name() -> str:
    return datetime.now().strftime("Session_%Y%m%d_%H%M%S")


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


def _resolve_skeleton_overlay_url(frame: Dict[str, Any]) -> str:
    skeleton_overlay_url = _first_present(
        frame,
        "highest_risk_image_url",
        "image",
        "skeleton_overlay_url",
        "image_url",
    )

    if skeleton_overlay_url and (
        "127.0.0.1" in str(skeleton_overlay_url) or "localhost" in str(skeleton_overlay_url)
    ):
        skeleton_overlay_url = ""

    return skeleton_overlay_url or ""


def _bulk_insert_risk_frames(
    conn: Any,
    session_id: int,
    risk_frames: List[Dict[str, Any]],
) -> None:
    for batch_start in range(0, len(risk_frames), RISK_FRAMES_BATCH_SIZE):
        batch = risk_frames[batch_start : batch_start + RISK_FRAMES_BATCH_SIZE]

        values_clauses = []
        params: Dict[str, Any] = {"session_id": session_id}
        jsonb_bindparams = []

        for idx, frame in enumerate(batch):
            skeleton_overlay_url = _resolve_skeleton_overlay_url(frame)

            values_clauses.append(
                f"(:session_id, :frame_number_{idx}, :risk_percentage_{idx}, "
                f":skeleton_overlay_url_{idx}, :joint_coordinates_{idx})"
            )
            params[f"frame_number_{idx}"] = _first_present(frame, "frame_number", "frame")
            params[f"risk_percentage_{idx}"] = _first_present(
                frame, "risk_percentage", "risk", "risk_score"
            )
            params[f"skeleton_overlay_url_{idx}"] = skeleton_overlay_url
            params[f"joint_coordinates_{idx}"] = _resolve_joint_coordinates(
                frame, skeleton_overlay_url
            )
            jsonb_bindparams.append(bindparam(f"joint_coordinates_{idx}", type_=JSONB))

        insert_sql = (
            "INSERT INTO risk_frames "
            "(session_id, frame_number, risk_percentage, skeleton_overlay_url, joint_coordinates) "
            "VALUES " + ", ".join(values_clauses) +
            " RETURNING frame_id"
        )

        result = conn.execute(
            text(insert_sql).bindparams(*jsonb_bindparams),
            params,
        )

        inserted_rows = result.mappings().all()
        if len(inserted_rows) != len(batch):
            raise SaveSessionFailedException()


def save_analysis_result(
    user_id: int,
    session_name: str,
    reference_video_id: int,
    video_user_url: str,
    accuracy_score: float,
    risk_frames: List[Dict[str, Any]],
    feedback: Dict[str, Any],
) -> Dict[str, Any]:
    session_name = (
        session_name.strip()
        if session_name and session_name.strip()
        else _generate_default_session_name()
    )
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
                raise SaveSessionFailedException()

            session_id = session_row["session_id"]

            if risk_frames:
                _bulk_insert_risk_frames(conn, session_id, risk_frames)

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
                raise SaveSessionFailedException()

        return {
            "success": True,
            "session_id": session_id,
            "session_name": session_name,
        }

    except SaveSessionFailedException:
        _cleanup_partial_save(session_id)
        raise
    except Exception:
        logger.exception("Failed to save analysis result")
        _cleanup_partial_save(session_id)
        raise SaveSessionFailedException()


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