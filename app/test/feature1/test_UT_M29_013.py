import pytest
from unittest.mock import patch

from app.services.feedback_service import generate_advanced_feedback
from app.exceptions import ServiceException


ANALYSIS_DATA = {
    "frame": 3,
    "time": 0.1,
    "angles": {
        "left_knee": 173.4,
        "right_knee": 162.2
    },
    "risk_score": 47.76
}

MOCK_FEEDBACK = {
    "form_summary": "Good squat depth.",
    "injury_risk": "Low risk.",
    "corrective_cues": "Keep knees over toes.",
    "practice_plan": "Practice 3 sets daily."
}

PROMPT_TEMPLATE = "Required keys: form_summary, injury_risk, corrective_cues, practice_plan"
MODEL_NAME = "glm-4.7-flash"


class TestGenerateAdvancedFeedback:

    @patch("app.services.feedback_service.parse_feedback_response")
    @patch("app.services.feedback_service._generate_with_ai")
    def test_returns_prompt_and_complete_feedback_when_api_succeeds(
        self,
        mock_generate,
        mock_parse
    ):
        mock_generate.return_value = (
            '{"form_summary": "Good squat depth.", '
            '"injury_risk": "Low risk.", '
            '"corrective_cues": "Keep knees over toes.", '
            '"practice_plan": "Practice 3 sets daily."}'
        )
        mock_parse.return_value = (MOCK_FEEDBACK, True)

        result = generate_advanced_feedback(
            ANALYSIS_DATA,
            PROMPT_TEMPLATE,
            MODEL_NAME
        )

        assert "prompt" in result
        assert "feedback" in result
        assert isinstance(result["prompt"], str)

        assert result["feedback"]["form_summary"] == "Good squat depth."
        assert result["feedback"]["injury_risk"] == "Low risk."
        assert result["feedback"]["corrective_cues"] == "Keep knees over toes."
        assert result["feedback"]["practice_plan"] == "Practice 3 sets daily."

        mock_generate.assert_called_once_with(
            result["prompt"],
            MODEL_NAME
        )

    @patch("app.services.feedback_service._generate_with_ai")
    def test_raises_service_exception_when_api_fails(
        self,
        mock_generate
    ):
        mock_generate.side_effect = ServiceException(
            "AI service failed to process the request."
        )

        with pytest.raises(ServiceException) as exc_info:
            generate_advanced_feedback(
                ANALYSIS_DATA,
                PROMPT_TEMPLATE,
                MODEL_NAME
            )

        assert "AI service failed to process the request." in str(exc_info.value)