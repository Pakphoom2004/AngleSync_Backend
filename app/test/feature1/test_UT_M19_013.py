import pytest

from app.services.feedback_service import build_feedback_prompt


ANALYSIS_DATA = {
    "frame": 3,
    "time": 0.1,
    "angles": {"left_knee": 173.4, "right_knee": 162.2},
    "risk_score": 47.76
}

PROMPT_TEMPLATE = "You are an expert fitness coach."


class TestBuildFeedbackPrompt:

    def test_returns_complete_prompt_string(self):
        result = build_feedback_prompt(ANALYSIS_DATA, PROMPT_TEMPLATE)

        assert isinstance(result, str)
        assert "You are an expert fitness coach." in result
        assert str(ANALYSIS_DATA)                in result