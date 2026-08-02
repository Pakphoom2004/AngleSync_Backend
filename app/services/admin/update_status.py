from typing import Dict, List, Any

from supabase import Client

from app.exceptions.status_update_failed_exception import StatusUpdateFailedException


def update_user_status(
    supabase: Client,
    user_id: int,
    target_user_id: int,
    new_status: str,
) -> Dict[str, object]:
    if user_id == target_user_id:
        raise StatusUpdateFailedException()

    try:
        response = (
            supabase.table("users")
            .update({"status": new_status})
            .eq("id", target_user_id)
            .execute()
        )
    except Exception:
        raise StatusUpdateFailedException()

    if not response.data:
        raise StatusUpdateFailedException()

    return {
        "success": True,
        "message": "User status updated successfully.",
    }

def list_users(supabase: Client) -> List[Dict[str, Any]]:
    response = (
        supabase.table("users")
        .select("user_id, username, user_role, user_status")
        .order("username")
        .execute()
    )
    return response.data or []