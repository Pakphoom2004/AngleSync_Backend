from typing import Any, Dict, List, Optional
from datetime import date, timedelta

from sqlalchemy import text

from app.config.db import get_connection
from app.exceptions.session_not_found_exception import SessionNotFoundException
from app.exceptions.unauthorized_access_exception import UnauthorizedAccessException

ADMINISTRATOR_ROLE = "Admin"


def session_list(
    user_id: int,
    sort_order: str,
    filter_date: Optional[date] = None,
) -> List[Dict[str, Any]]:
    ascending = sort_order == "asc"
    order_clause = "ASC" if ascending else "DESC"

    with get_connection() as conn:
        # user_id ที่ส่งเข้ามาคือผู้เรียก (Admin ที่ login) ไม่ใช่ user
        # ที่จะเอา session มาแสดง — endpoint นี้คืน session ของทุก user
        # ในระบบ (system-wide) เพื่อให้ตรงกับตัวเลข "Total Analysis
        # Sessions" ใน dashboard_summary จึงต้องเช็ค role ก่อนเสมอ
        user_row = conn.execute(
            text("SELECT user_role FROM users WHERE user_id = :user_id LIMIT 1"),
            {"user_id": user_id},
        ).mappings().first()

        if not user_row or user_row.get("user_role") != ADMINISTRATOR_ROLE:
            raise UnauthorizedAccessException()

        query = "SELECT * FROM analysis_sessions"
        params: Dict[str, Any] = {}

        if filter_date is not None:
            query += " WHERE analysis_date >= :start_of_day AND analysis_date < :end_of_day"
            params["start_of_day"] = filter_date.isoformat()
            params["end_of_day"] = (filter_date + timedelta(days=1)).isoformat()

        query += f" ORDER BY analysis_date {order_clause}"

        rows = conn.execute(text(query), params).mappings().all()

    sessions = [dict(row) for row in rows]

    if filter_date is not None and not sessions:
        raise SessionNotFoundException()

    return sessions