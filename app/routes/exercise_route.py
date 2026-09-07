from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text

from app.config.db import get_connection
from app.services.auth_service import get_current_user_id, user_by_id

router = APIRouter()


@router.get("/exercises")
def get_exercises(user_id: int = Depends(get_current_user_id)):
    user = user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found."
        )

    user_gender = user.get("gender")

    with get_connection() as conn:
        if user_gender and user_gender.lower() in ["male", "female"]:
            query = text(
                """
                SELECT reference_video_id, exercise_name, reference_video_url, reference_gender
                FROM exercise_reference
                WHERE LOWER(reference_gender) = LOWER(:gender)
                """
            )
            rows = conn.execute(query, {"gender": user_gender}).mappings().all()
        else:
            query = text(
                """
                SELECT reference_video_id, exercise_name, reference_video_url, reference_gender
                FROM exercise_reference
                """
            )
            rows = conn.execute(query).mappings().all()

    return [dict(row) for row in rows]