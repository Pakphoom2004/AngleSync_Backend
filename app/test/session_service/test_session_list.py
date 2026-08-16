import pytest
from datetime import date
from unittest.mock import MagicMock, patch
from app.services.admin.session_service import session_list
from app.exceptions.session_not_found_exception import SessionNotFoundException

SESSIONS = [
    {"session_id": 1, "session_name": "Session A", "analysis_date": "2026-07-01T10:00:00"},
    {"session_id": 2, "session_name": "Session B", "analysis_date": "2026-07-15T10:00:00"},
    {"session_id": 3, "session_name": "Session C", "analysis_date": "2026-07-20T10:00:00"},
]


def _mock_connection(return_data):
    mock_conn = MagicMock()
    result = MagicMock()
    result.mappings.return_value.all.return_value = return_data
    mock_conn.execute.return_value = result
    return mock_conn


def _patch_get_connection(mock_conn):
    mock_get_connection = MagicMock()
    mock_get_connection.return_value.__enter__.return_value = mock_conn
    mock_get_connection.return_value.__exit__.return_value = False
    return patch(
        "app.services.admin.session_service.get_connection",
        mock_get_connection,
    )


# UT-01
def test_session_list_sort_asc_and_desc():
    asc_data = sorted(SESSIONS, key=lambda s: s["analysis_date"])
    mock_conn_asc = _mock_connection(asc_data)

    with _patch_get_connection(mock_conn_asc):
        result_asc = session_list(user_id=1, sort_order="asc", filter_date=None)

    assert [s["analysis_date"] for s in result_asc] == [
        "2026-07-01T10:00:00",
        "2026-07-15T10:00:00",
        "2026-07-20T10:00:00",
    ]

    desc_data = sorted(SESSIONS, key=lambda s: s["analysis_date"], reverse=True)
    mock_conn_desc = _mock_connection(desc_data)

    with _patch_get_connection(mock_conn_desc):
        result_desc = session_list(user_id=1, sort_order="desc", filter_date=None)

    assert [s["analysis_date"] for s in result_desc] == [
        "2026-07-20T10:00:00",
        "2026-07-15T10:00:00",
        "2026-07-01T10:00:00",
    ]


# UT-02
def test_session_list_filter_date_returns_matching_session_only():
    filtered_data = [s for s in SESSIONS if s["analysis_date"].startswith("2026-07-15")]
    mock_conn = _mock_connection(filtered_data)

    with _patch_get_connection(mock_conn):
        result = session_list(
            user_id=1,
            sort_order="desc",
            filter_date=date(2026, 7, 15),
        )

    assert len(result) == 1
    assert result[0]["analysis_date"] == "2026-07-15T10:00:00"


# UT-03
def test_session_list_raises_when_no_sessions_match_filter_date():
    mock_conn = _mock_connection([])

    with _patch_get_connection(mock_conn):
        with pytest.raises(SessionNotFoundException) as exc_info:
            session_list(
                user_id=1,
                sort_order="desc",
                filter_date=date(2026, 1, 1),
            )

    assert str(exc_info.value) == "No sessions found for the selected date."