from typing import Dict
from supabase import Client
from app.exceptions.unauthorized_access_exception import UnauthorizedAccessException

ADMINISTRATOR_ROLE = "Admin"

def dashboard_summary(supabase: Client, user_id: int) -> Dict[str, int]:
    try:
        user_response = (
            supabase.table("users")
            .select("user_role")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
    except Exception:
        raise UnauthorizedAccessException()

    if not user_response.data or user_response.data[0].get("user_role") != ADMINISTRATOR_ROLE:
        raise UnauthorizedAccessException()

    try:
        users_count_response = (
            supabase.table("users").select("user_id", count="exact").execute()
        )
        sessions_count_response = (
            supabase.table("analysis_sessions").select("session_id", count="exact").execute()
        )
    except Exception:
        raise UnauthorizedAccessException()

    total_users = users_count_response.count or 0
    total_analysis_sessions = sessions_count_response.count or 0

    return {
        "total_users": total_users,
        "total_analysis_sessions": total_analysis_sessions,
    }