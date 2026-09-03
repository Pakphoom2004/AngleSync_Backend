import pytest
from unittest.mock import patch
from app.services.auth_service import verify_google_id_token
from app.exceptions.auth_exception import AuthException


@patch("google.oauth2.id_token.verify_oauth2_token")
@patch("google.auth.transport.requests.Request")
def test_verify_google_id_token_success(mock_request, mock_verify):
    expected_payload = {
        "sub": "1234567890",
        "email": "user@example.com",
        "name": "Test User",
        "picture": "https://example.com/photo.jpg"
    }
    mock_verify.return_value = expected_payload

    result = verify_google_id_token("valid_token_abc123")

    assert result == expected_payload
    assert result["sub"] == "1234567890"
    assert result["email"] == "user@example.com"
    assert result["name"] == "Test User"
    assert result["picture"] == "https://example.com/photo.jpg"


@patch("google.oauth2.id_token.verify_oauth2_token")
def test_verify_google_id_token_expired(mock_verify):
    mock_verify.side_effect = ValueError("Token expired")

    with pytest.raises(AuthException) as exc_info:
        verify_google_id_token("invalid_malformed_string")

    assert str(exc_info.value) == "Authentication failed."


@patch("google.oauth2.id_token.verify_oauth2_token")
def test_verify_google_id_token_malformed(mock_verify):
    mock_verify.side_effect = ValueError("Wrong number of segments in token")

    with pytest.raises(AuthException) as exc_info:
        verify_google_id_token("invalid_malformed_string")

    assert str(exc_info.value) == "Authentication failed."


@patch("google.oauth2.id_token.verify_oauth2_token")
def test_verify_google_id_token_different_client_id(mock_verify):
    mock_verify.side_effect = ValueError("Token's audience doesn't match client ID")

    with pytest.raises(AuthException) as exc_info:
        verify_google_id_token("other_client_id_token")

    assert str(exc_info.value) == "Authentication failed."


@patch("google.oauth2.id_token.verify_oauth2_token")
def test_verify_google_id_token_empty(mock_verify):
    mock_verify.side_effect = ValueError("Token is empty")

    with pytest.raises(AuthException) as exc_info:
        verify_google_id_token("")

    assert str(exc_info.value) == "Authentication failed."