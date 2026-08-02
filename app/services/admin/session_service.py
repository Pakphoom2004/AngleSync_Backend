from typing import Any, Dict, List, Optional
from datetime import date, timedelta

from supabase import Client

from app.exceptions.session_not_found_exception import SessionNotFoundException


def session_list(
    supabase: Client,
    user_id: int,
    sort_order: str,
    filter_date: Optional[date] = None,
) -> List[Dict[str, Any]]:
    query = supabase.table("analysis_sessions").select("*").eq("user_id", user_id)

    if filter_date is not None:
        start_of_day = filter_date.isoformat()
        end_of_day = (filter_date + timedelta(days=1)).isoformat()
        query = query.gte("analysis_date", start_of_day).lt("analysis_date", end_of_day)

    ascending = sort_order == "asc"
    query = query.order("analysis_date", desc=not ascending)

    response = query.execute()
    sessions = response.data or []

    if filter_date is not None and not sessions:
        raise SessionNotFoundException()

    return sessions