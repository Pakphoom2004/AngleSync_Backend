from typing import Dict

from sqlalchemy import text

from app.config.db import get_connection
from app.exceptions.unauthorized_access_exception import UnauthorizedAccessException

ADMINISTRATOR_ROLE = "Admin"

def dashboard_summary(user_id: int) -> Dict[str, int]:
    try:
        with get_connection() as conn:
            user_row = conn.execute(
                text("SELECT user_role FROM users WHERE user_id = :user_id LIMIT 1"),
                {"user_id": user_id},
            ).mappings().first()

            if not user_row or user_row.get("user_role") != ADMINISTRATOR_ROLE:
                raise UnauthorizedAccessException()

            total_users = conn.execute(text("SELECT COUNT(*) FROM users")).scalar_one()
            total_analysis_sessions = conn.execute(
                text("SELECT COUNT(*) FROM analysis_sessions")
            ).scalar_one()
    except UnauthorizedAccessException:
        raise
    except Exception:
        raise UnauthorizedAccessException()

    return {
        "total_users": total_users,
        "total_analysis_sessions": total_analysis_sessions,
    }