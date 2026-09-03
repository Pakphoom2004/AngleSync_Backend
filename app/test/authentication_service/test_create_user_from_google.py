import pytest
from unittest.mock import patch, MagicMock

from app.services.auth_service import create_user_from_google


@patch("app.services.auth_service.get_connection")
def test_create_user_from_google_existing_user(mock_get_connection):

    google_payload = {
        "sub": "10000000000000000001",
        "email": "existing_user@example.com",
        "name": "Existing User",
        "picture": "https://example.com/avatar.jpg"
    }

    expected_user = {
        "user_id": 1,
        "username": "Existing User",
        "email": "existing_user@example.com",
        "google_sub": "10000000000000000001",
        "profile_picture_url": "https://example.com/avatar.jpg",
        "user_role": "Member",
        "user_status": "Active"
    }

    mock_conn = MagicMock()
    mock_get_connection.return_value.__enter__.return_value = mock_conn

    mock_mappings = MagicMock()
    mock_mappings.fetchone.return_value = expected_user
    mock_conn.execute.return_value.mappings.return_value = mock_mappings

    result = create_user_from_google(google_payload)

    assert result == expected_user
    assert result["google_sub"] == google_payload["sub"]
    assert result["email"] == google_payload["email"]

    assert mock_conn.execute.call_count == 1


@patch("app.services.auth_service.get_connection")
def test_create_user_from_google_new_user(mock_get_connection):

    google_payload = {
        "sub": "20000000000000000002",
        "email": "new_user@example.com",
        "name": "New User",
        "picture": "https://example.com/avatar2.jpg"
    }

    inserted_user = {
        "user_id": 2,
        "username": "New User",
        "email": "new_user@example.com",
        "google_sub": "20000000000000000002",
        "profile_picture_url": "https://example.com/avatar2.jpg",
        "user_role": "Member",
        "user_status": "Active"
    }

    mock_conn = MagicMock()
    mock_get_connection.return_value.__enter__.return_value = mock_conn

    mock_select_mappings = MagicMock()
    mock_select_mappings.fetchone.return_value = None

    mock_insert_mappings = MagicMock()
    mock_insert_mappings.fetchone.return_value = inserted_user

    mock_conn.execute.return_value.mappings.side_effect = [
        mock_select_mappings,
        mock_insert_mappings
    ]

    result = create_user_from_google(google_payload)

    assert result == inserted_user
    assert result["user_role"] == "Member"
    assert result["user_status"] == "Active"

    assert mock_conn.execute.call_count == 2