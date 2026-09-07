from datetime import date
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.services.admin.dashboard_service import dashboard_summary
from app.services.admin.session_service import session_list
from app.services.admin.update_status import update_user_status, list_users

from app.exceptions.unauthorized_access_exception import UnauthorizedAccessException
from app.exceptions.session_not_found_exception import SessionNotFoundException
from app.exceptions.status_update_failed_exception import StatusUpdateFailedException

router = APIRouter(prefix="/admin")


@router.get("/dashboard-summary")
async def get_dashboard_summary(user_id: int = Query(...)):
    try:
        return dashboard_summary(user_id)
    except UnauthorizedAccessException as error:
        raise HTTPException(status_code=403, detail=str(error))


@router.get("/sessions")
async def get_sessions(
    user_id: int = Query(...),
    sort_order: str = Query("desc"),
    filter_date: Optional[date] = Query(None),
):
    try:
        sessions = session_list(user_id, sort_order, filter_date)
        return {"sessions": sessions}
    except SessionNotFoundException as error:
        raise HTTPException(status_code=404, detail=str(error))


@router.get("/users")
async def get_users():
    users = list_users()
    return {"users": users}


class UpdateUserStatusRequest(BaseModel):
    user_id: int
    target_user_id: int
    new_status: str


@router.post("/update-user-status")
async def post_update_user_status(payload: UpdateUserStatusRequest):
    try:
        return update_user_status(
            payload.user_id,
            payload.target_user_id,
            payload.new_status,
        )
    except StatusUpdateFailedException as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail="Internal server error")