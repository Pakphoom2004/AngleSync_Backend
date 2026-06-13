import pytest

from app.services.feedback_service import parse_feedback_response


class TestParseFeedbackResponse:

    def test_parses_valid_json_response(self):
        text = (
            '{"form_summary": "Good squat depth.", '
            '"injury_risk": "Low risk.", '
            '"corrective_cues": "Keep knees over toes.", '
            '"practice_plan": "Practice 3 sets daily."}'
        )

        feedback, is_complete = parse_feedback_response(text)

        assert feedback["form_summary"]    == "Good squat depth."
        assert feedback["injury_risk"]     == "Low risk."
        assert feedback["corrective_cues"] == "Keep knees over toes."
        assert feedback["practice_plan"]   == "Practice 3 sets daily."
        assert is_complete is True

    def test_is_complete_false_when_field_is_empty(self):
        text = (
            '{"form_summary": "Good squat depth.", '
            '"injury_risk": "", '
            '"corrective_cues": "Keep knees over toes.", '
            '"practice_plan": "Practice 3 sets daily."}'
        )

        feedback, is_complete = parse_feedback_response(text)

        assert feedback["injury_risk"] == ""
        assert is_complete is False