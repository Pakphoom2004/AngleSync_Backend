from fastapi import APIRouter
from sqlalchemy import text

from app.config.db import get_connection

router = APIRouter()


@router.get("/exercises")
def get_exercises():
    with get_connection() as conn:
        rows = conn.execute(
            text(
                """
                SELECT reference_video_id, exercise_name, reference_video_url, reference_gender
                FROM exercise_reference
                """
            )
        ).mappings().all()

    return [dict(row) for row in rows]
