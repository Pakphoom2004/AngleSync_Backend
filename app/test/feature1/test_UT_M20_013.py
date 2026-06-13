import pytest
from unittest.mock import MagicMock, patch

from app.services.feedback_service import _generate_with_zai
from app.exceptions import ServiceException


class TestGenerateWithZai:

    @patch("app.services.feedback_service.OpenAI")
    @patch("app.services.feedback_service.os.getenv")
    def test_returns_response_text_when_api_succeeds(
        self, mock_getenv, mock_openai_cls
    ):
        mock_getenv.return_value = "fake-api-key"

        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            '{"form_summary": "Good form.", '
            '"injury_risk": "Low.", '
            '"corrective_cues": "Keep knees over toes.", '
            '"practice_plan": "Practice daily."}'
        )
        mock_openai_cls.return_value.chat.completions.create.return_value = (
            mock_response
        )

        result = _generate_with_zai(
            "You are an expert fitness coach.",
            "squat_frame_150.jpg",
            "gemini-2.0-flash"
        )

        assert isinstance(result, str)
        assert len(result) > 0

    @patch("app.services.feedback_service.OpenAI")
    @patch("app.services.feedback_service.os.getenv")
    def test_raises_service_exception_when_api_fails(
        self, mock_getenv, mock_openai_cls
    ):
        mock_getenv.return_value = "fake-api-key"
        mock_openai_cls.return_value.chat.completions.create.side_effect = (
            Exception("API error")
        )

        with pytest.raises(ServiceException):
            _generate_with_zai(
                "You are an expert fitness coach.",
                "squat_frame_150.jpg",
                "gemini-2.0-flash"
            )