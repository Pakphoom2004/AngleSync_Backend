from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from app.exceptions.auth_exception import AuthException
from app.services.auth_service import (
    complete_profile,
    create_access_token,
    create_user_from_google,
    decode_access_token,
    verify_google_id_token,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


class GoogleLoginRequest(BaseModel):
    id_token: str


class CompleteProfileRequest(BaseModel):
    gender: str


def get_current_user_id(authorization: str = Header(...)) -> int:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token scheme.")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        return decode_access_token(token)
    except AuthException as error:
        raise HTTPException(status_code=401, detail=str(error))


@router.post("/google")
async def login_with_google(payload: GoogleLoginRequest):
    try:
        google_payload = verify_google_id_token(payload.id_token)
        user = create_user_from_google(google_payload)
        access_token = create_access_token(user["user_id"])   # ⬅️ เอา ._mapping ออก

        return {
            "access_token": access_token,
            "user_id": user["user_id"],       # ⬅️ เอา ._mapping ออก
            "username": user["username"],     # ⬅️ เอา ._mapping ออก
            "email": user["email"],           # ⬅️ เอา ._mapping ออก
            "needs_gender": user.get("gender") is None,
        }
    except AuthException as error:
        raise HTTPException(status_code=401, detail=str(error))


@router.post("/complete-profile")
async def complete_profile_route(
    payload: CompleteProfileRequest,
    user_id: int = Depends(get_current_user_id),
):
    try:
        user = complete_profile(user_id, payload.gender)
        return {
            "user_id": user["user_id"],
            "gender": user["gender"],
        }
    except AuthException as error:
        raise HTTPException(status_code=400, detail=str(error))