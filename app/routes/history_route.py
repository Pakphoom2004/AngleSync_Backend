from datetime import date
from typing import Literal, Optional
from fastapi import APIRouter, HTTPException, Query, Depends

from app.exceptions.history_exception import HistoryException
from app.services.history_user.history_service import (
    history_list,
    session_detail,
    delete_session, filter_and_sort_history_by_date,
)
from app.exceptions.session_delete_failed_exception import SessionDeleteFailedException
from app.services.auth_service import get_current_user_id

router = APIRouter()


@router.get("/history")
async def get_history(
    search_term: Optional[str] = Query(None),
    sort_order: Literal["asc", "desc"] = Query("desc"),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    user_id: int = Depends(get_current_user_id),
):
    try:
        # หากมีการส่งวันที่มา ให้ใช้ filter_and_sort_history_by_date
        if start_date or end_date:
            history = filter_and_sort_history_by_date(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
                sort_order=sort_order,
            )
        else:
            history = history_list(
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
    user_id: int = Depends(get_current_user_id), # ⬅️ ถอด user_id จาก Bearer token
):
    try:
        return session_detail(
            user_id=user_id,
            session_id=session_id,
        )
    except HistoryException as error:
        raise HTTPException(status_code=404, detail=str(error))


@router.delete("/history/{session_id}")
async def delete_history_session(
    session_id: int,
    user_id: int = Depends(get_current_user_id), # ⬅️ ถอด user_id จาก Bearer token
):
    try:
        return delete_session(
            user_id=user_id,
            session_id=session_id,
        )
    except SessionDeleteFailedException as error:
        raise HTTPException(status_code=400, detail=str(error))