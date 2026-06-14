import pytest
from unittest.mock import patch

from app.services.feedback_service import generate_advanced_feedback
from app.exceptions import ServiceException


ANALYSIS_DATA = {
    "frame": 3,
    "time": 0.1,
    "angles": {"left_knee": 173.4, "right_knee": 162.2},
    "risk_score": 47.76
}

MOCK_RESPONSE = (
    '{"form_summary": "Good squat depth.", '
    '"injury_risk": "Low risk.", '
    '"corrective_cues": "Keep knees over toes.", '
    '"practice_plan": "Practice 3 sets daily."}'
)


class TestGenerateAdvancedFeedback:

    @patch("app.services.feedback_service._generate_with_zai")
    def test_returns_prompt_and_complete_feedback_when_api_succeeds(
        self, mock_generate
    ):
        mock_generate.return_value = MOCK_RESPONSE

        result = generate_advanced_feedback(
            ANALYSIS_DATA,
            "squat_frame_150.jpg"
        )

        assert "prompt"   in result
        assert "feedback" in result
        assert isinstance(result["prompt"], str)

        feedback = result["feedback"]
        assert feedback["form_summary"]    != ""
        assert feedback["injury_risk"]     != ""
        assert feedback["corrective_cues"] != ""
        assert feedback["practice_plan"]   != ""

    @patch("app.services.feedback_service._generate_with_zai")
    def test_frame_path_does_not_replace_prompt_template(
        self, mock_generate
    ):
        mock_generate.return_value = MOCK_RESPONSE

        result = generate_advanced_feedback(
            ANALYSIS_DATA,
            "squat_frame_150.jpg"
        )

        assert "Required keys:" in result["prompt"]
        assert "squat_frame_150.jpg" not in result["prompt"]

    @patch("app.services.feedback_service._generate_with_zai")
    def test_raises_service_exception_when_api_fails(
        self, mock_generate
    ):
        mock_generate.side_effect = ServiceException(
            "AI service failed to process the request."
        )

        with pytest.raises(ServiceException) as exc_info:
            generate_advanced_feedback(
                ANALYSIS_DATA,
                "squat_frame_150.jpg"
            )

        assert "AI service failed to process the request." in str(exc_info.value)
