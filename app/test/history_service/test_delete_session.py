import pytest
from unittest.mock import MagicMock, call
from app.services.history_user import delete_session
from app.exceptions.session_delete_failed_exception import SessionDeleteFailedException


def _make_supabase_mock(
    session_data=None,
    session_raises=None,
    delete_session_data=None,
    delete_raises=None,
):
    mock_supabase = MagicMock()
    call_count = {"n": 0}

    def table_side_effect(table_name):
        builder = MagicMock()
        builder.select.return_value = builder
        builder.delete.return_value = builder
        builder.eq.return_value = builder
        builder.limit.return_value = builder

        n = call_count["n"]
        call_count["n"] += 1

        if table_name == "analysis_sessions" and n == 0:
            if session_raises:
                builder.execute.side_effect = session_raises
            else:
                result = MagicMock()
                result.data = session_data
                builder.execute.return_value = result

        elif table_name == "risk_frames":
            builder.execute.return_value = MagicMock()

        elif table_name == "feedbacks":
            builder.execute.return_value = MagicMock()

        elif table_name == "analysis_sessions" and n > 0:
            if delete_raises:
                builder.execute.side_effect = delete_raises
            else:
                result = MagicMock()
                result.data = delete_session_data
                builder.execute.return_value = result

        return builder

    mock_supabase.table.side_effect = table_side_effect
    return mock_supabase


# UT-01
def test_delete_session_success():
    mock_supabase = _make_supabase_mock(
        session_data=[{"session_id": 101}],
        delete_session_data=[{"session_id": 101}],
    )

    result = delete_session(mock_supabase, user_id=1, session_id=101)

    assert result["success"] is True

    called_tables = [c.args[0] for c in mock_supabase.table.call_args_list]
    assert "risk_frames" in called_tables
    assert "feedbacks" in called_tables
    assert called_tables.count("analysis_sessions") >= 2


# UT-02
def test_delete_session_raises_when_session_not_found():
    mock_supabase = _make_supabase_mock(
        session_data=[],
    )

    with pytest.raises(SessionDeleteFailedException) as exc_info:
        delete_session(mock_supabase, user_id=1, session_id=999)

    assert str(exc_info.value) == "Unable to delete this record. Please try again."

    called_tables = [c.args[0] for c in mock_supabase.table.call_args_list]
    assert "risk_frames" not in called_tables
    assert "feedbacks" not in called_tables


# UT-03
def test_delete_session_raises_when_delete_operation_fails():
    mock_supabase = _make_supabase_mock(
        session_data=[{"session_id": 101}],
        delete_raises=Exception("DB connection error"),
    )

    with pytest.raises(SessionDeleteFailedException) as exc_info:
        delete_session(mock_supabase, user_id=1, session_id=101)

    assert str(exc_info.value) == "Unable to delete this record. Please try again."