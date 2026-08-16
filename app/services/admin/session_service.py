from typing import Any, Dict, List, Optional
from datetime import date, timedelta

from sqlalchemy import text

from app.config.db import get_connection
from app.exceptions.session_not_found_exception import SessionNotFoundException


def session_list(
    user_id: int,
    sort_order: str,
    filter_date: Optional[date] = None,
) -> List[Dict[str, Any]]:
    ascending = sort_order == "asc"
    order_clause = "ASC" if ascending else "DESC"

    query = "SELECT * FROM analysis_sessions WHERE user_id = :user_id"
    params: Dict[str, Any] = {"user_id": user_id}

    if filter_date is not None:
        query += " AND analysis_date >= :start_of_day AND analysis_date < :end_of_day"
        params["start_of_day"] = filter_date.isoformat()
        params["end_of_day"] = (filter_date + timedelta(days=1)).isoformat()

    query += f" ORDER BY analysis_date {order_clause}"

    with get_connection() as conn:
        rows = conn.execute(text(query), params).mappings().all()

    sessions = [dict(row) for row in rows]

    if filter_date is not None and not sessions:
        raise SessionNotFoundException()

    return sessions