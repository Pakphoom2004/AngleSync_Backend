import jwt
import pytest
from unittest.mock import patch
import datetime

from app.services.auth_service import create_access_token, JWT_SECRET_KEY, JWT_ALGORITHM

def test_create_access_token_success():
    user_id = 1

    token = create_access_token(user_id)

    assert isinstance(token, str)

    decoded_payload = jwt.decode(
        token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM]
    )

    assert decoded_payload["sub"] == "1"
    assert "exp" in decoded_payload
    assert isinstance(decoded_payload["exp"], int)


def test_create_access_token_different_user_id():
    user_id = 999

    token = create_access_token(user_id)

    assert isinstance(token, str)

    decoded_payload = jwt.decode(
        token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM]
    )

    assert decoded_payload["sub"] == "999"
    assert isinstance(decoded_payload["sub"], str)
    assert "exp" in decoded_payload

def test_create_access_token_expiration_one_hour():
    user_id = 1

    token = create_access_token(user_id)

    decoded_payload = jwt.decode(
        token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM]
    )

    exp_timestamp = decoded_payload["exp"]
    now_timestamp = datetime.datetime.now(datetime.timezone.utc).timestamp()

    time_difference = exp_timestamp - now_timestamp
    assert 3590 <= time_difference <= 3600