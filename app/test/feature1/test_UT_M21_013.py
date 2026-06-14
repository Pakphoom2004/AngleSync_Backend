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

    def test_parses_feedback_wrapped_json_response(self):
        text = (
            '{"feedback": {'
            '"form_summary": "Good squat depth.", '
            '"injury_risk": "Low risk.", '
            '"corrective_cues": "Keep knees over toes.", '
            '"practice_plan": "Practice 3 sets daily."'
            '}}'
        )

        feedback, is_complete = parse_feedback_response(text)

        assert feedback["form_summary"]    == "Good squat depth."
        assert feedback["injury_risk"]     == "Low risk."
        assert feedback["corrective_cues"] == "Keep knees over toes."
        assert feedback["practice_plan"]   == "Practice 3 sets daily."
        assert is_complete is True

    def test_parses_markdown_model_response(self):
        text = """
Based on the analysis data provided for frame 162, here is the detailed assessment.

### **Overall Risk Assessment: HIGH RISK**
**Risk Score:** 43.43

The posture detected indicates a significant deviation from neutral body mechanics.

### **Detailed Posture Analysis**

**1. Lateral Flexion (Leaning)**
*   **The Issue:** There is a distinct curve in the spine.
*   **Impact:** This lateral twist places strain on the spine.

### **Recommended Correction**
1.  **Level the Shoulders:** Bring the right shoulder down.
2.  **Reduce Elbow Flexion:** Straighten the right elbow slightly.
3.  **Center the Spine:** Rotate the torso back to neutral.
"""

        feedback, is_complete = parse_feedback_response(text)

        assert "Detailed Posture Analysis" in feedback["form_summary"]
        assert "HIGH RISK" in feedback["injury_risk"]
        assert "Level the Shoulders" in feedback["corrective_cues"]
        assert "Center the Spine" in feedback["practice_plan"]
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
