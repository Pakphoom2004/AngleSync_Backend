from typing import Dict

from sqlalchemy import text

from app.config.db import get_connection
from app.exceptions.unauthorized_access_exception import UnauthorizedAccessException
from app.exceptions.data_load_exception import DataLoadException

ADMINISTRATOR_ROLE = "Admin"

def dashboard_summary(user_id: int) -> Dict[str, int]:
    with get_connection() as conn:
        try:
            user_row = conn.execute(
                text("SELECT user_role FROM users WHERE user_id = :user_id LIMIT 1"),
                {"user_id": user_id},
            ).mappings().first()
        except Exception:
            raise DataLoadException()

        if not user_row or user_row.get("user_role") != ADMINISTRATOR_ROLE:
            raise UnauthorizedAccessException()

        try:
            total_users = conn.execute(text("SELECT COUNT(*) FROM users")).scalar_one()
            total_analysis_sessions = conn.execute(
                text("SELECT COUNT(*) FROM analysis_sessions")
            ).scalar_one()
        except Exception:
            raise DataLoadException()

    return {
        "total_users": total_users,
        "total_analysis_sessions": total_analysis_sessions,
    }