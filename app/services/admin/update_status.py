from typing import Dict, List, Any

from sqlalchemy import text

from app.config.db import get_connection
from app.exceptions.status_update_failed_exception import StatusUpdateFailedException


def update_user_status(
    user_id: int,
    target_user_id: int,
    new_status: str,
) -> Dict[str, object]:
    if user_id == target_user_id:
        raise StatusUpdateFailedException()

    try:
        with get_connection() as conn:
            result = conn.execute(
                text(
                    """
                    UPDATE users
                    SET user_status = :new_status
                    WHERE user_id = :target_user_id
                    RETURNING user_id
                    """
                ),
                {"new_status": new_status, "target_user_id": target_user_id},
            )
            updated = result.mappings().all()
    except Exception:
        raise StatusUpdateFailedException()

    if not updated:
        raise StatusUpdateFailedException()

    return {
        "success": True,
        "message": "User status updated successfully.",
    }

def list_users() -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            text(
                """
                SELECT user_id, username, user_role, user_status
                FROM users
                ORDER BY username
                """
            )
        ).mappings().all()
    return [dict(row) for row in rows]