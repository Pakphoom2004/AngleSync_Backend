import pytest
from datetime import date
from unittest.mock import MagicMock
from app.services.admin.session_service import session_list
from app.exceptions.session_not_found_exception import SessionNotFoundException

SESSIONS = [
    {"session_id": 1, "session_name": "Session A", "analysis_date": "2026-07-01T10:00:00"},
    {"session_id": 2, "session_name": "Session B", "analysis_date": "2026-07-15T10:00:00"},
    {"session_id": 3, "session_name": "Session C", "analysis_date": "2026-07-20T10:00:00"},
]


def _make_supabase_mock(return_data):
    mock_supabase = MagicMock()
    result = MagicMock()
    result.data = return_data

    builder = MagicMock()
    builder.select.return_value = builder
    builder.eq.return_value = builder
    builder.gte.return_value = builder
    builder.lt.return_value = builder
    builder.order.return_value = builder
    builder.execute.return_value = result

    mock_supabase.table.return_value = builder
    return mock_supabase


# UT-01
def test_session_list_sort_asc_and_desc():
    asc_data = sorted(SESSIONS, key=lambda s: s["analysis_date"])
    mock_supabase_asc = _make_supabase_mock(asc_data)

    result_asc = session_list(mock_supabase_asc, user_id=1, sort_order="asc", filter_date=None)

    assert [s["analysis_date"] for s in result_asc] == [
        "2026-07-01T10:00:00",
        "2026-07-15T10:00:00",
        "2026-07-20T10:00:00",
    ]

    desc_data = sorted(SESSIONS, key=lambda s: s["analysis_date"], reverse=True)
    mock_supabase_desc = _make_supabase_mock(desc_data)

    result_desc = session_list(mock_supabase_desc, user_id=1, sort_order="desc", filter_date=None)

    assert [s["analysis_date"] for s in result_desc] == [
        "2026-07-20T10:00:00",
        "2026-07-15T10:00:00",
        "2026-07-01T10:00:00",
    ]


# UT-02
def test_session_list_filter_date_returns_matching_session_only():
    filtered_data = [s for s in SESSIONS if s["analysis_date"].startswith("2026-07-15")]
    mock_supabase = _make_supabase_mock(filtered_data)

    result = session_list(
        mock_supabase,
        user_id=1,
        sort_order="desc",
        filter_date=date(2026, 7, 15),
    )

    assert len(result) == 1
    assert result[0]["analysis_date"] == "2026-07-15T10:00:00"


# UT-03
def test_session_list_raises_when_no_sessions_match_filter_date():
    mock_supabase = _make_supabase_mock(return_data=[])

    with pytest.raises(SessionNotFoundException) as exc_info:
        session_list(
            mock_supabase,
            user_id=1,
            sort_order="desc",
            filter_date=date(2026, 1, 1),
        )

    assert str(exc_info.value) == "No sessions found for the selected date."