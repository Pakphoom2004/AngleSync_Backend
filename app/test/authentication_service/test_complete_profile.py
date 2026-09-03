import pytest
from unittest.mock import patch, MagicMock
from app.services.auth_service import complete_profile
from app.exceptions.auth_exception import AuthException


@patch("app.services.auth_service.get_connection")
def test_complete_profile_success_male(mock_get_connection):
    user_id = 1
    gender = "Male"
    expected_user = {
        "user_id": 1,
        "username": "Test User",
        "email": "test@example.com",
        "gender": "Male",
        "user_role": "Member",
        "user_status": "Active"
    }

    mock_conn = MagicMock()
    mock_get_connection.return_value.__enter__.return_value = mock_conn
    mock_conn.execute.return_value.mappings.return_value.first.return_value = expected_user

    result = complete_profile(user_id, gender)

    assert result == expected_user
    assert result["gender"] == "Male"
    mock_conn.commit.assert_called_once()


@patch("app.services.auth_service.get_connection")
def test_complete_profile_success_female(mock_get_connection):
    user_id = 1
    gender = "Female"
    expected_user = {
        "user_id": 1,
        "username": "Test User",
        "email": "test@example.com",
        "gender": "Female",
        "user_role": "Member",
        "user_status": "Active"
    }

    mock_conn = MagicMock()
    mock_get_connection.return_value.__enter__.return_value = mock_conn
    mock_conn.execute.return_value.mappings.return_value.first.return_value = expected_user

    result = complete_profile(user_id, gender)

    assert result == expected_user
    assert result["gender"] == "Female"
    mock_conn.commit.assert_called_once()


def test_complete_profile_invalid_gender():
    with pytest.raises(AuthException) as exc_info:
        complete_profile(user_id=1, gender="Other")

    assert str(exc_info.value) == "Failed to complete profile."


@patch("app.services.auth_service.get_connection")
def test_complete_profile_user_not_found(mock_get_connection):
    mock_conn = MagicMock()
    mock_get_connection.return_value.__enter__.return_value = mock_conn
    mock_conn.execute.return_value.mappings.return_value.first.return_value = None

    with pytest.raises(AuthException) as exc_info:
        complete_profile(user_id=0, gender="Male")

    assert str(exc_info.value) == "Failed to complete profile."