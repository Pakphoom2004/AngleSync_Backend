import pytest
from unittest.mock import MagicMock, patch
from app.services.history_user.history_service import history_list
from app.exceptions.history_exception import HistoryException

SESSIONS = [
    {"session_id": 1, "session_name": "Squat Session", "analysis_date": "2026-07-01T10:00:00"},
    {"session_id": 2, "session_name": "Lunge Session", "analysis_date": "2026-07-15T10:00:00"},
    {"session_id": 3, "session_name": "Squat Practice", "analysis_date": "2026-07-20T10:00:00"},
]


def _mock_connection(return_data=None, raises=None):
    mock_conn = MagicMock()

    if raises:
        mock_conn.execute.side_effect = raises
    else:
        result = MagicMock()
        result.mappings.return_value.all.return_value = return_data or []
        mock_conn.execute.return_value = result

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
def test_history_list_sort_asc_and_desc():
    asc_data = sorted(SESSIONS, key=lambda s: s["analysis_date"])
    mock_conn_asc = _mock_connection(asc_data)

    with _patch_get_connection(mock_conn_asc):
        result_asc = history_list(user_id=1, search_term=None, sort_order="asc")

    assert [s["analysis_date"] for s in result_asc] == [
        "2026-07-01T10:00:00",
        "2026-07-15T10:00:00",
        "2026-07-20T10:00:00",
    ]

    desc_data = sorted(SESSIONS, key=lambda s: s["analysis_date"], reverse=True)
    mock_conn_desc = _mock_connection(desc_data)

    with _patch_get_connection(mock_conn_desc):
        result_desc = history_list(user_id=1, search_term=None, sort_order="desc")

    assert [s["analysis_date"] for s in result_desc] == [
        "2026-07-20T10:00:00",
        "2026-07-15T10:00:00",
        "2026-07-01T10:00:00",
    ]


# UT-02
def test_history_list_search_term_filters_by_session_name():
    filtered_data = [
        s for s in SESSIONS
        if "squat" in s["session_name"].lower()
    ]
    mock_conn = _mock_connection(filtered_data)

    with _patch_get_connection(mock_conn):
        result = history_list(user_id=1, search_term="Squat", sort_order="desc")

    assert len(result) == 2
    names = [s["session_name"] for s in result]
    assert "Squat Session" in names
    assert "Squat Practice" in names
    assert "Lunge Session" not in names


# UT-03
def test_history_list_raises_when_db_query_fails():
    mock_conn = _mock_connection(raises=Exception("DB connection error"))

    with _patch_get_connection(mock_conn):
        with pytest.raises(HistoryException) as exc_info:
            history_list(user_id=1, search_term=None, sort_order="desc")

    assert str(exc_info.value) == "Couldn't load your history right now. Please try again."