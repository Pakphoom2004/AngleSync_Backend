from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query

from app.exceptions.history_exception import HistoryException
from app.services.history_user.history_service import history_list, session_detail


router = APIRouter()


def get_supabase_client():
    from app.config.supabase_client import supabase

    return supabase


@router.get("/history")
async def get_history(
    user_id: int = Query(1),
    search_term: Optional[str] = Query(None),
    sort_order: Literal["asc", "desc"] = Query("desc"),
):
    try:
        history = history_list(
            get_supabase_client(),
            user_id=user_id,
            search_term=search_term,
            sort_order=sort_order,
        )
        return {"history": history}
    except HistoryException as error:
        raise HTTPException(status_code=404, detail=str(error))


@router.get("/history/{session_id}")
async def get_session_detail(
    session_id: int,
    user_id: int = Query(1),
):
    try:
        return session_detail(
            get_supabase_client(),
            user_id=user_id,
            session_id=session_id,
        )
    except HistoryException as error:
        raise HTTPException(status_code=404, detail=str(error))
