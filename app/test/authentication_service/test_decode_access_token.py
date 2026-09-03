import jwt
import pytest
from unittest.mock import patch
from app.services.auth_service import decode_access_token
from app.exceptions.auth_exception import AuthException


@patch("jwt.decode")
def test_decode_access_token_success(mock_jwt_decode):
    """UT Case 1: Verify that the function successfully decodes a valid JWT access token and returns user_id as integer."""
    mock_jwt_decode.return_value = {"sub": "1"}

    user_id = decode_access_token("valid_jwt_token_for_user_1")

    assert user_id == 1
    assert isinstance(user_id, int)


@patch("jwt.decode")
def test_decode_access_token_expired(mock_jwt_decode):
    """UT Case 2: Verify that the function raises AuthException when the token has passed its expiration time."""
    mock_jwt_decode.side_effect = jwt.ExpiredSignatureError("Signature has expired")

    with pytest.raises(AuthException) as exc_info:
        decode_access_token("expired_jwt_token")

    assert str(exc_info.value) == "Invalid or expired token."


@patch("jwt.decode")
def test_decode_access_token_tampered(mock_jwt_decode):
    """UT Case 3: Verify that the function raises AuthException when the token signature is invalid or tampered with."""
    mock_jwt_decode.side_effect = jwt.InvalidSignatureError("Signature verification failed")

    with pytest.raises(AuthException) as exc_info:
        decode_access_token("tampered_jwt_token")

    assert str(exc_info.value) == "Invalid or expired token."


@patch("jwt.decode")
def test_decode_access_token_malformed(mock_jwt_decode):
    """UT Case 4: Verify that the function raises AuthException when the token is malformed or invalid format."""
    mock_jwt_decode.side_effect = jwt.DecodeError("Not enough segments")

    with pytest.raises(AuthException) as exc_info:
        decode_access_token("invalid_token_string")

    assert str(exc_info.value) == "Invalid or expired token."


@patch("jwt.decode")
def test_decode_access_token_empty(mock_jwt_decode):
    """UT Case 5: Verify that the function raises AuthException when the token parameter is empty."""
    mock_jwt_decode.side_effect = jwt.DecodeError("Not enough segments")

    with pytest.raises(AuthException) as exc_info:
        decode_access_token("")

    assert str(exc_info.value) == "Invalid or expired token."