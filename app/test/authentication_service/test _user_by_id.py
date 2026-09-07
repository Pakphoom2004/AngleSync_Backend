import pytest
from unittest.mock import MagicMock, patch
from app.exceptions.auth_exception import AuthException
from app.services.auth_service import user_by_id

@patch("app.services.auth_service.get_connection")
def test_get_user_by_id_active_user_success(mock_get_connection):
    # Mock Database Connection และ Query Result
    mock_conn = MagicMock()
    mock_get_connection.return_value.__enter__.return_value = mock_conn
    mock_conn.execute.return_value.mappings.return_value.first.return_value = {
        "user_id": 1,
        "username": "John Doe",
        "email": "john@example.com",
        "user_role": "Member",
        "user_status": "Active",
    }

    result = user_by_id(1)

    assert result is not None
    assert result["user_id"] == 1
    assert result["username"] == "John Doe"
    assert result["user_status"] == "Active"

@patch("app.services.auth_service.get_connection")
def test_get_user_by_id_not_found_returns_none(mock_get_connection):
    mock_conn = MagicMock()
    mock_get_connection.return_value.__enter__.return_value = mock_conn
    mock_conn.execute.return_value.mappings.return_value.first.return_value = None

    result = user_by_id(999)

    assert result is None

@patch("app.services.auth_service.get_connection")
def test_get_user_by_id_suspended_user_raises_exception(mock_get_connection):
    mock_conn = MagicMock()
    mock_get_connection.return_value.__enter__.return_value = mock_conn
    mock_conn.execute.return_value.mappings.return_value.first.return_value = {
        "user_id": 2,
        "username": "Jane Doe",
        "email": "jane@example.com",
        "user_role": "Member",
        "user_status": "Suspended",
    }
    with pytest.raises(AuthException) as exc_info:
        user_by_id(2)

    assert str(exc_info.value) == "Your account has been suspended. Please contact support for assistance."