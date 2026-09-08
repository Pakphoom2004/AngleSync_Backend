from datetime import date
from unittest.mock import MagicMock, patch
import pytest

from app.exceptions.history_exception import HistoryException
from app.exceptions.session_delete_failed_exception import SessionDeleteFailedException
from app.services.history_user.history_service import (
    delete_session,
    filter_and_sort_history_by_date,
    history_list,
    search_history_by_session_name,
    session_detail,
)

# ==========================================
# UT-01: history_list
# ==========================================
class TestHistoryList:

    @patch('app.services.history_user.history_service.get_connection')
    def test_history_list_sort_order(self, mock_get_connection):
        # UT-01-01: Verify sorting directions
        mock_conn = MagicMock()
        mock_get_connection.return_value.__enter__.return_value = mock_conn

        mock_rows = [
            {"session_id": 1, "session_name": "S1", "analysis_date": "2026-07-20"},
            {"session_id": 2, "session_name": "S2", "analysis_date": "2026-07-15"},
        ]
        mock_conn.execute.return_value.mappings.return_value.all.return_value = mock_rows

        result = history_list(user_id=1, search_term=None, sort_order="desc")

        assert len(result) == 2
        assert result[0]["analysis_date"] == "2026-07-20"
        mock_conn.execute.assert_called_once()

    @patch('app.services.history_user.history_service.get_connection')
    def test_history_list_filter_search_term(self, mock_get_connection):
        # UT-01-02: Verify search term filtering
        mock_conn = MagicMock()
        mock_get_connection.return_value.__enter__.return_value = mock_conn

        mock_rows = [
            {"session_id": 1, "session_name": "Squat Session"},
            {"session_id": 3, "session_name": "Squat Practice"},
        ]
        mock_conn.execute.return_value.mappings.return_value.all.return_value = mock_rows

        result = history_list(user_id=1, search_term="Squat", sort_order="desc")

        assert len(result) == 2
        assert "Squat" in result[0]["session_name"]

    @patch('app.services.history_user.history_service.get_connection')
    def test_history_list_exception(self, mock_get_connection):
        # UT-01-03: Verify exception handling
        mock_get_connection.side_effect = Exception("Database connection error")

        with pytest.raises(HistoryException):
            history_list(user_id=1, search_term=None, sort_order="desc")


# ==========================================
# UT-02: search_history_by_session_name
# ==========================================
class TestSearchHistoryBySessionName:

    @patch('app.services.history_user.history_service.history_list')
    def test_search_history_with_keyword(self, mock_history_list):
        # UT-02-01: Verify searching with keyword
        mock_history_list.return_value = [{"session_id": 1, "session_name": "Squat Session"}]

        result = search_history_by_session_name(user_id=1, search_term="squat")

        assert len(result) == 1
        assert result[0]["session_name"] == "Squat Session"
        mock_history_list.assert_called_once_with(user_id=1, search_term="squat", sort_order="desc")

    @patch('app.services.history_user.history_service.history_list')
    def test_search_history_empty_term_returns_all(self, mock_history_list):
        # UT-02-02: Verify empty search term
        mock_history_list.return_value = [
            {"session_id": 1, "session_name": "Squat Session"},
            {"session_id": 2, "session_name": "Lunge Session"},
        ]

        result = search_history_by_session_name(user_id=1, search_term="")

        assert len(result) == 2
        mock_history_list.assert_called_once_with(user_id=1, search_term="", sort_order="desc")

    @patch('app.services.history_user.history_service.history_list')
    def test_search_history_no_match(self, mock_history_list):
        # UT-02-03: Verify non-existing keyword
        mock_history_list.return_value = []

        result = search_history_by_session_name(user_id=1, search_term="UnknownExercise")

        assert result == []
        mock_history_list.assert_called_once_with(user_id=1, search_term="UnknownExercise", sort_order="desc")

    @patch('app.services.history_user.history_service.history_list')
    def test_search_history_exception(self, mock_history_list):
        # UT-02-04: Verify exception on search failure
        mock_history_list.side_effect = HistoryException()

        with pytest.raises(HistoryException):
            search_history_by_session_name(user_id=1, search_term="Squat")

# ==========================================
# UT-03: filter_and_sort_history_by_date
# ==========================================
class TestFilterAndSortHistoryByDate:

    @patch('app.services.history_user.history_service.get_connection')
    def test_filter_by_date_range(self, mock_get_connection):
        # UT-03-01: Verify date range filter
        mock_conn = MagicMock()
        mock_get_connection.return_value.__enter__.return_value = mock_conn

        mock_rows = [{"session_id": 1, "session_name": "Squat", "analysis_date": "2026-07-05"}]
        mock_conn.execute.return_value.mappings.return_value.all.return_value = mock_rows

        result = filter_and_sort_history_by_date(
            user_id=1,
            sort_order="desc",
            start_date=date(2026, 7, 1),
            end_date=date(2026, 7, 15),
        )

        assert len(result) == 1

    @patch('app.services.history_user.history_service.get_connection')
    def test_sort_by_date_ascending(self, mock_get_connection):
        # UT-03-02: Verify ascending sort order
        mock_conn = MagicMock()
        mock_get_connection.return_value.__enter__.return_value = mock_conn

        mock_rows = [
            {"session_id": 1, "analysis_date": "2026-07-01"},
            {"session_id": 2, "analysis_date": "2026-07-15"},
        ]
        mock_conn.execute.return_value.mappings.return_value.all.return_value = mock_rows

        result = filter_and_sort_history_by_date(user_id=1, sort_order="asc")

        assert result[0]["analysis_date"] == "2026-07-01"

    @patch('app.services.history_user.history_service.get_connection')
    def test_filter_and_sort_exception(self, mock_get_connection):
        # UT-03-03: Verify DB exception
        mock_get_connection.side_effect = Exception("DB error")

        with pytest.raises(HistoryException):
            filter_and_sort_history_by_date(user_id=1, sort_order="desc", start_date=date(2026, 7, 1))


# ==========================================
# UT-04: session_detail
# ==========================================
class TestSessionDetail:

    @patch('app.services.history_user.history_service.get_connection')
    def test_session_detail_success(self, mock_get_connection):
        # UT-04-01: Verify returning complete session details
        mock_conn = MagicMock()
        mock_get_connection.return_value.__enter__.return_value = mock_conn

        # Mock Session Query
        mock_session = {
            "session_name": "Squat Session",
            "video_user_url": "https://storage.example.com/video.mp4",
            "accuracy_score": 85.5,
            "reference_video_id": 10,
        }
        # Mock Risk Frames Query
        mock_risk_frames = [
            {
                "frame_id": 1,
                "session_id": 101,
                "frame_number": 12,
                "risk_percentage": 70.0,
                "skeleton_overlay_url": "https://storage.example.com/frame.jpg",
                "joint_coordinates": {},
            }
        ]
        # Mock Feedback Query
        mock_feedback = {
            "form_summary": "Good form",
            "injury_risk": "Low",
            "corrective_cues": "Keep back straight",
            "practice_plan": "3 sets of 10 reps",
        }

        mock_conn.execute.return_value.mappings.side_effect = [
            MagicMock(first=lambda: mock_session),
            MagicMock(all=lambda: mock_risk_frames),
            MagicMock(first=lambda: mock_feedback),
        ]

        result = session_detail(user_id=1, session_id=101)

        assert result["session_id"] == 101
        assert result["session_name"] == "Squat Session"
        assert result["video_user_url"] == "https://storage.example.com/video.mp4"
        assert len(result["analysis_result"]["risk_frames"]) == 1

    @patch('app.services.history_user.history_service.get_connection')
    def test_session_detail_video_unavailable_fallback(self, mock_get_connection):
        # UT-04-02: Verify video URL fallback when missing
        mock_conn = MagicMock()
        mock_get_connection.return_value.__enter__.return_value = mock_conn

        mock_session = {
            "session_name": "Squat Session",
            "video_user_url": None,
            "accuracy_score": 80.0,
            "reference_video_id": 10,
        }

        mock_conn.execute.return_value.mappings.side_effect = [
            MagicMock(first=lambda: mock_session),
            MagicMock(all=lambda: []),
            MagicMock(first=lambda: None),
        ]

        result = session_detail(user_id=1, session_id=102)

        assert result["video_user_url"] == "Video not available."

    @patch('app.services.history_user.history_service.get_connection')
    def test_session_detail_not_found_raises_exception(self, mock_get_connection):
        # UT-04-03: Verify exception when session_id not found
        mock_conn = MagicMock()
        mock_get_connection.return_value.__enter__.return_value = mock_conn

        mock_conn.execute.return_value.mappings.return_value.first.return_value = None

        with pytest.raises(HistoryException):
            session_detail(user_id=1, session_id=999)


# ==========================================
# UT-05: delete_session
# ==========================================
class TestDeleteSession:

    @patch('app.services.history_user.history_service.get_connection')
    def test_delete_session_success(self, mock_get_connection):
        # UT-05-01: Verify successful deletion
        mock_conn = MagicMock()
        mock_get_connection.return_value.__enter__.return_value = mock_conn

        mock_session = {"session_id": 101}
        mock_deleted_rows = [{"session_id": 101}]

        mock_conn.execute.return_value.mappings.side_effect = [
            MagicMock(first=lambda: mock_session),
            MagicMock(all=lambda: mock_deleted_rows),
        ]

        result = delete_session(user_id=1, session_id=101)

        assert result == {"success": True}

    @patch('app.services.history_user.history_service.get_connection')
    def test_delete_session_not_found(self, mock_get_connection):
        # UT-05-02: Verify Exception on non-existent session
        mock_conn = MagicMock()
        mock_get_connection.return_value.__enter__.return_value = mock_conn

        mock_conn.execute.return_value.mappings.return_value.first.return_value = None

        with pytest.raises(SessionDeleteFailedException):
            delete_session(user_id=1, session_id=999)