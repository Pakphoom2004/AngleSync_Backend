from typing import Optional
from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel

from app.exceptions.auth_exception import AuthException
from app.services.auth_service import (
    complete_profile,
    create_access_token,
    create_user_from_google,
    decode_access_token,
    user_by_id,
    verify_google_id_token,
    get_current_user_id,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


class GoogleLoginRequest(BaseModel):
    id_token: str


class CompleteProfileRequest(BaseModel):
    gender: Optional[str] = "Unspecified" # รองรับ Optional และ Default เป็น Unspecified


@router.post("/google")
async def login_with_google(payload: GoogleLoginRequest):
    try:
        google_payload = verify_google_id_token(payload.id_token)
        user = create_user_from_google(google_payload)
        access_token = create_access_token(user["user_id"])

        return {
            "access_token": access_token,
            "user_id": user["user_id"],
            "username": user["username"],
            "email": user["email"],
            "user_role": user["user_role"],
            "gender": user.get("gender"),
            # ถ้า gender เป็น None ให้ถือว่าต้องระบุโปรไฟล์ก่อน
            "needs_gender": user.get("gender") is None,
        }
    except AuthException as error:
        if "suspended" in str(error).lower():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=str(error),
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=str(error)
        )

@router.get("/me")
async def get_current_user(user_id: int = Depends(get_current_user_id)):
    try:
        user = user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found."
            )

        return {
            "user_id": user["user_id"],
            "username": user["username"],
            "email": user["email"],
            "user_role": user["user_role"],
            "gender": user.get("gender"),
            "needs_gender": user.get("gender") is None,
        }
    except AuthException as error:
        if "suspended" in str(error).lower():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=str(error)
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=str(error)
        )

@router.post("/complete-profile")
async def complete_profile_route(
    payload: CompleteProfileRequest,
    user_id: int = Depends(get_current_user_id),
):
    try:
        gender_value = payload.gender if payload.gender else "Unspecified"
        user = complete_profile(user_id, gender_value)
        return {
            "user_id": user["user_id"],
            "gender": user["gender"],
        }
    except AuthException as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)
        )