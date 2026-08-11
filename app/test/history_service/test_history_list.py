import pytest
from unittest.mock import MagicMock
from app.services.history_service import history_list
from app.exceptions.history_exception import HistoryException

SESSIONS = [
    {"session_id": 1, "session_name": "Squat Session", "analysis_date": "2026-07-01T10:00:00"},
    {"session_id": 2, "session_name": "Lunge Session", "analysis_date": "2026-07-15T10:00:00"},
    {"session_id": 3, "session_name": "Squat Practice", "analysis_date": "2026-07-20T10:00:00"},
]


def _make_supabase_mock(return_data=None, raises=None):
    mock_supabase = MagicMock()
    builder = MagicMock()

    builder.select.return_value = builder
    builder.eq.return_value = builder
    builder.ilike.return_value = builder
    builder.order.return_value = builder

    if raises:
        builder.execute.side_effect = raises
    else:
        result = MagicMock()
        result.data = return_data
        builder.execute.return_value = result

    mock_supabase.table.return_value = builder
    return mock_supabase


# UT-01
def test_history_list_sort_asc_and_desc():
    asc_data = sorted(SESSIONS, key=lambda s: s["analysis_date"])
    mock_supabase_asc = _make_supabase_mock(asc_data)

    result_asc = history_list(mock_supabase_asc, user_id=1, search_term=None, sort_order="asc")

    assert [s["analysis_date"] for s in result_asc] == [
        "2026-07-01T10:00:00",
        "2026-07-15T10:00:00",
        "2026-07-20T10:00:00",
    ]

    desc_data = sorted(SESSIONS, key=lambda s: s["analysis_date"], reverse=True)
    mock_supabase_desc = _make_supabase_mock(desc_data)

    result_desc = history_list(mock_supabase_desc, user_id=1, search_term=None, sort_order="desc")

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
    mock_supabase = _make_supabase_mock(filtered_data)

    result = history_list(mock_supabase, user_id=1, search_term="Squat", sort_order="desc")

    assert len(result) == 2
    names = [s["session_name"] for s in result]
    assert "Squat Session" in names
    assert "Squat Practice" in names
    assert "Lunge Session" not in names


# UT-03
def test_history_list_raises_when_db_query_fails():
    mock_supabase = _make_supabase_mock(raises=Exception("DB connection error"))

    with pytest.raises(HistoryException) as exc_info:
        history_list(mock_supabase, user_id=1, search_term=None, sort_order="desc")

    assert exc_info.value.message == "Couldn't load your history right now. Please try again."