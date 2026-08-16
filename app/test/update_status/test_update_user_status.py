import pytest
from unittest.mock import MagicMock, patch
from app.services.admin.update_status import update_user_status
from app.exceptions.status_update_failed_exception import StatusUpdateFailedException


def _mock_connection(updated_rows=None, raises=None):
    mock_conn = MagicMock()

    if raises:
        mock_conn.execute.side_effect = raises
    else:
        result = MagicMock()
        result.mappings.return_value.all.return_value = updated_rows or []
        mock_conn.execute.return_value = result

    return mock_conn


def _patch_get_connection(mock_conn):
    mock_get_connection = MagicMock()
    mock_get_connection.return_value.__enter__.return_value = mock_conn
    mock_get_connection.return_value.__exit__.return_value = False
    return patch(
        "app.services.admin.update_status.get_connection",
        mock_get_connection,
    )


# UT-01
def test_update_user_status_success():
    mock_conn = _mock_connection(
        updated_rows=[{"user_id": 5}]
    )

    with _patch_get_connection(mock_conn):
        result = update_user_status(
            user_id=1,
            target_user_id=5,
            new_status="Inactive",
        )

    assert result["success"] is True
    assert result["message"] == "User status updated successfully."


# UT-02
def test_update_user_status_raises_when_admin_deactivates_self():
    mock_conn = _mock_connection()

    with _patch_get_connection(mock_conn):
        with pytest.raises(StatusUpdateFailedException) as exc_info:
            update_user_status(
                user_id=1,
                target_user_id=1,
                new_status="Inactive",
            )

    assert str(exc_info.value) == "Unable to update user status. Please try again."
    mock_conn.execute.assert_not_called()


# UT-03
def test_update_user_status_raises_when_db_update_fails():
    mock_conn = _mock_connection(
        raises=Exception("DB connection error")
    )

    with _patch_get_connection(mock_conn):
        with pytest.raises(StatusUpdateFailedException) as exc_info:
            update_user_status(
                user_id=1,
                target_user_id=5,
                new_status="Inactive",
            )

    assert str(exc_info.value) == "Unable to update user status. Please try again."