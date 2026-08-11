import pytest
from unittest.mock import MagicMock
from app.services.user_management_service import update_user_status
from app.exceptions.status_update_failed_exception import StatusUpdateFailedException


def _make_supabase_mock(return_data=None, raises=None):
    mock_supabase = MagicMock()
    builder = MagicMock()

    if raises:
        builder.update.return_value.eq.return_value.execute.side_effect = raises
    else:
        result = MagicMock()
        result.data = return_data
        builder.update.return_value.eq.return_value.execute.return_value = result

    mock_supabase.table.return_value = builder
    return mock_supabase


# UT-01
def test_update_user_status_success():
    mock_supabase = _make_supabase_mock(
        return_data=[{"user_id": 5, "user_status": "Inactive"}]
    )

    result = update_user_status(
        supabase=mock_supabase,
        user_id=1,
        target_user_id=5,
        new_status="Inactive",
    )

    assert result["success"] is True
    assert result["message"] == "User status updated successfully."


# UT-02
def test_update_user_status_raises_when_admin_deactivates_self():
    mock_supabase = _make_supabase_mock()

    with pytest.raises(StatusUpdateFailedException) as exc_info:
        update_user_status(
            supabase=mock_supabase,
            user_id=1,
            target_user_id=1,
            new_status="Inactive",
        )

    assert str(exc_info.value) == "Unable to update user status. Please try again."
    mock_supabase.table.assert_not_called()


# UT-03
def test_update_user_status_raises_when_db_update_fails():
    mock_supabase = _make_supabase_mock(
        raises=Exception("DB connection error")
    )

    with pytest.raises(StatusUpdateFailedException) as exc_info:
        update_user_status(
            supabase=mock_supabase,
            user_id=1,
            target_user_id=5,
            new_status="Inactive",
        )

    assert str(exc_info.value) == "Unable to update user status. Please try again."