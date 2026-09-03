from __future__ import annotations

import datetime
from typing import Any, Dict

import jwt
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from sqlalchemy import text

from app.config.auth_config import (
    GOOGLE_CLIENT_ID,
    JWT_ALGORITHM,
    JWT_EXPIRE_MINUTES,
    JWT_SECRET_KEY,
)
from app.config.db import get_connection
from app.exceptions.auth_exception import AuthException
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def verify_google_id_token(token: str) -> Dict[str, Any]:
    try:
        payload = id_token.verify_oauth2_token(
            token, google_requests.Request(), GOOGLE_CLIENT_ID
        )
        return payload
    except ValueError:
        raise AuthException("Authentication failed.")


def create_user_from_google(google_payload):
    email = google_payload.get("email")
    google_sub = google_payload.get("sub")

    with get_connection() as conn:
        # 1. เช็กว่ามี user หรือยัง (ใช้ .mappings() เพื่อให้ดึงค่าแบบ dict ได้)
        existing_user = conn.execute(
            text("SELECT * FROM users WHERE google_sub = :sub OR email = :email"),
            {"sub": google_sub, "email": email}
        ).mappings().fetchone()

        if existing_user:
            return existing_user

        # 2. ถ้ายังไม่มี ให้ทำการ INSERT ผู้ใช้ใหม่
        inserted = conn.execute(
            text("""
                INSERT INTO users
                    (username, email, google_sub, profile_picture_url, user_role, user_status)
                VALUES
                    (:username, :email, :google_sub, :profile_picture_url, 'Member', 'Active')
                RETURNING *
            """),
            {
                "username": google_payload.get("name"),
                "email": email,
                "google_sub": google_sub,
                "profile_picture_url": google_payload.get("picture"),
            }
        ).mappings().fetchone()

        return inserted

def complete_profile(user_id: int, gender: str) -> Dict[str, Any]:
    if gender not in ("Male", "Female"):
        raise AuthException("Failed to complete profile.")

    with get_connection() as conn:
        updated = conn.execute(
            text(
                """
                UPDATE users
                SET gender = :gender
                WHERE user_id = :user_id
                RETURNING *
                """
            ),
            {"gender": gender, "user_id": user_id},
        ).mappings().first()

        if not updated:
            raise AuthException("Failed to complete profile.")

        conn.commit()
        return dict(updated)


def create_access_token(user_id: int) -> str:
    expire = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
        minutes=JWT_EXPIRE_MINUTES
    )
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def decode_access_token(token: str) -> int:
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return int(payload["sub"])
    except jwt.PyJWTError:
        raise AuthException("Invalid or expired token.")

def get_user_by_id(user_id: int) -> Dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            text("SELECT * FROM users WHERE user_id = :user_id LIMIT 1"),
            {"user_id": user_id},
        ).mappings().first()
        return dict(row) if row else None

def get_current_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)) -> int:
        try:
            # credentials.credentials จะได้ค่า token จาก Header "Authorization: Bearer <token>"
            return decode_access_token(credentials.credentials)
        except AuthException as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e),
                headers={"WWW-Authenticate": "Bearer"},
            )