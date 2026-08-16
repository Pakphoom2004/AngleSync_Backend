import pytest
from unittest.mock import MagicMock, patch
from app.services.history_user.history_service import delete_session
from app.exceptions.session_delete_failed_exception import SessionDeleteFailedException


def _mock_connection(
    session_row=None,
    session_raises=None,
    deleted_rows=None,
    delete_raises=None,
):
    mock_conn = MagicMock()
    calls = {"n": 0}

    def execute_side_effect(*args, **kwargs):
        n = calls["n"]
        calls["n"] += 1

        if n == 0:
            if session_raises:
                raise session_raises
            result = MagicMock()
            result.mappings.return_value.first.return_value = session_row
            return result

        if n in (1, 2):
            return MagicMock()

        if delete_raises:
            raise delete_raises

        result = MagicMock()
        result.mappings.return_value.all.return_value = deleted_rows or []
        return result

    mock_conn.execute.side_effect = execute_side_effect
    return mock_conn


def _patch_get_connection(mock_conn):
    mock_get_connection = MagicMock()
    mock_get_connection.return_value.__enter__.return_value = mock_conn
    mock_get_connection.return_value.__exit__.return_value = False
    return patch(
        "app.services.history_user.history_service.get_connection",
        mock_get_connection,
    )


# UT-01
def test_delete_session_success():
    mock_conn = _mock_connection(
        session_row={"session_id": 101},
        deleted_rows=[{"session_id": 101}],
    )

    with _patch_get_connection(mock_conn):
        result = delete_session(user_id=1, session_id=101)

    assert result["success"] is True
    assert mock_conn.execute.call_count == 4


# UT-02
def test_delete_session_raises_when_session_not_found():
    mock_conn = _mock_connection(session_row=None)

    with _patch_get_connection(mock_conn):
        with pytest.raises(SessionDeleteFailedException) as exc_info:
            delete_session(user_id=1, session_id=999)

    assert str(exc_info.value) == "Unable to delete this record. Please try again."
    assert mock_conn.execute.call_count == 1


# UT-03
def test_delete_session_raises_when_delete_operation_fails():
    mock_conn = _mock_connection(
        session_row={"session_id": 101},
        delete_raises=Exception("DB connection error"),
    )

    with _patch_get_connection(mock_conn):
        with pytest.raises(SessionDeleteFailedException) as exc_info:
            delete_session(user_id=1, session_id=101)

    assert str(exc_info.value) == "Unable to delete this record. Please try again."