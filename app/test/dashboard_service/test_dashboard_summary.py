import pytest
from unittest.mock import MagicMock, patch
from app.services.admin.dashboard_service import dashboard_summary
from app.exceptions.unauthorized_access_exception import UnauthorizedAccessException

ADMINISTRATOR_ROLE = "Admin"


def _mock_connection(
    user_role=ADMINISTRATOR_ROLE,
    user_query_raises=None,
    users_count=0,
    sessions_count=0,
    count_query_raises=None,
):
    mock_conn = MagicMock()
    calls = {"n": 0}

    def execute_side_effect(*args, **kwargs):
        n = calls["n"]
        calls["n"] += 1

        if n == 0:
            if user_query_raises:
                raise user_query_raises
            result = MagicMock()
            result.mappings.return_value.first.return_value = (
                {"user_role": user_role} if user_role else None
            )
            return result

        if count_query_raises:
            raise count_query_raises

        result = MagicMock()
        result.scalar_one.return_value = users_count if n == 1 else sessions_count
        return result

    mock_conn.execute.side_effect = execute_side_effect
    return mock_conn


def _patch_get_connection(mock_conn):
    mock_get_connection = MagicMock()
    mock_get_connection.return_value.__enter__.return_value = mock_conn
    mock_get_connection.return_value.__exit__.return_value = False
    return patch(
        "app.services.admin.dashboard_service.get_connection",
        mock_get_connection,
    )


# UT-01
def test_dashboard_summary_returns_correct_counts():
    mock_conn = _mock_connection(
        user_role=ADMINISTRATOR_ROLE,
        users_count=25,
        sessions_count=120,
    )

    with _patch_get_connection(mock_conn):
        result = dashboard_summary(user_id=1)

    assert result["total_users"] == 25
    assert result["total_analysis_sessions"] == 120


# UT-02
def test_dashboard_summary_returns_zero_when_no_data():
    mock_conn = _mock_connection(
        user_role=ADMINISTRATOR_ROLE,
        users_count=0,
        sessions_count=0,
    )

    with _patch_get_connection(mock_conn):
        result = dashboard_summary(user_id=1)

    assert result["total_users"] == 0
    assert result["total_analysis_sessions"] == 0


# UT-03
def test_dashboard_summary_raises_when_user_is_not_admin():
    mock_conn = _mock_connection(
        user_role="Member",
    )

    with _patch_get_connection(mock_conn):
        with pytest.raises(UnauthorizedAccessException) as exc_info:
            dashboard_summary(user_id=2)

    assert str(exc_info.value) == "Unable to process your request. Please try again."


# UT-04
def test_dashboard_summary_raises_when_count_query_fails():
    mock_conn = _mock_connection(
        user_role=ADMINISTRATOR_ROLE,
        count_query_raises=Exception("DB connection error"),
    )

    with _patch_get_connection(mock_conn):
        with pytest.raises(UnauthorizedAccessException) as exc_info:
            dashboard_summary(user_id=1)

    assert str(exc_info.value) == "Unable to process your request. Please try again."