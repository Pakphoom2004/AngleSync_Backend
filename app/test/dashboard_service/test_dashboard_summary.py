import pytest
from unittest.mock import MagicMock
from app.services.admin.dashboard_service import dashboard_summary
from app.exceptions.unauthorized_access_exception import UnauthorizedAccessException

ADMINISTRATOR_ROLE = "Admin"


def _make_supabase_mock(
    user_role=ADMINISTRATOR_ROLE,
    user_query_raises=None,
    users_count=None,
    sessions_count=None,
    count_query_raises=None,
):
    mock_supabase = MagicMock()
    call_count = {"n": 0}

    def table_side_effect(table_name):
        builder = MagicMock()
        n = call_count["n"]
        call_count["n"] += 1

        # 1st call: role check query
        if n == 0 and table_name == "users":
            if user_query_raises:
                builder.select.return_value.eq.return_value.limit.return_value.execute.side_effect = user_query_raises
            else:
                result = MagicMock()
                result.data = [{"user_role": user_role}] if user_role else []
                builder.select.return_value.eq.return_value.limit.return_value.execute.return_value = result

        # 2nd call: users count
        elif table_name == "users" and n > 0:
            if count_query_raises:
                builder.select.return_value.execute.side_effect = count_query_raises
            else:
                result = MagicMock()
                result.count = users_count
                builder.select.return_value.execute.return_value = result

        # 3rd call: sessions count
        elif table_name == "analysis_sessions":
            if count_query_raises:
                builder.select.return_value.execute.side_effect = count_query_raises
            else:
                result = MagicMock()
                result.count = sessions_count
                builder.select.return_value.execute.return_value = result

        return builder

    mock_supabase.table.side_effect = table_side_effect
    return mock_supabase


# UT-01
def test_dashboard_summary_returns_correct_counts():
    mock_supabase = _make_supabase_mock(
        user_role=ADMINISTRATOR_ROLE,
        users_count=25,
        sessions_count=120,
    )

    result = dashboard_summary(mock_supabase, user_id=1)

    assert result["total_users"] == 25
    assert result["total_analysis_sessions"] == 120


# UT-02
def test_dashboard_summary_returns_zero_when_no_data():
    mock_supabase = _make_supabase_mock(
        user_role=ADMINISTRATOR_ROLE,
        users_count=None,
        sessions_count=None,
    )

    result = dashboard_summary(mock_supabase, user_id=1)

    assert result["total_users"] == 0
    assert result["total_analysis_sessions"] == 0


# UT-03
def test_dashboard_summary_raises_when_user_is_not_admin():
    mock_supabase = _make_supabase_mock(
        user_role="Member",
    )

    with pytest.raises(UnauthorizedAccessException) as exc_info:
        dashboard_summary(mock_supabase, user_id=2)

    assert exc_info.value.message == "Unable to load data. Please try again."


# UT-04
def test_dashboard_summary_raises_when_count_query_fails():
    mock_supabase = _make_supabase_mock(
        user_role=ADMINISTRATOR_ROLE,
        count_query_raises=Exception("DB connection error"),
    )

    with pytest.raises(UnauthorizedAccessException) as exc_info:
        dashboard_summary(mock_supabase, user_id=1)

    assert exc_info.value.message == "Unable to load data. Please try again."