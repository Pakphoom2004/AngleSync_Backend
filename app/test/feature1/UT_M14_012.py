import pytest

from app.services.motion_analysis import analyze_motion
from app.exceptions import InvalidKeypointsException


KP01 = [
    {"id": 0,  "name": "nose",           "x": 0.4680759608745575,  "y": 0.13428252935409546, "confidence": 0.8291665315628052},
    {"id": 1,  "name": "left_eye",       "x": 0.0,                 "y": 0.0,                 "confidence": 0.20284414291381836},
    {"id": 2,  "name": "right_eye",      "x": 0.4652771055698395,  "y": 0.11990076303482056, "confidence": 0.8898438811302185},
    {"id": 3,  "name": "left_ear",       "x": 0.0,                 "y": 0.0,                 "confidence": 0.050865743309259415},
    {"id": 4,  "name": "right_ear",      "x": 0.43922296166419983, "y": 0.11978987604379654, "confidence": 0.9512590169906616},
    {"id": 5,  "name": "left_shoulder",  "x": 0.42056581377983093, "y": 0.2169395238161087,  "confidence": 0.9129444360733032},
    {"id": 6,  "name": "right_shoulder", "x": 0.4168631434440613,  "y": 0.22159205377101898, "confidence": 0.9958706498146057},
    {"id": 7,  "name": "left_elbow",     "x": 0.41852816939353943, "y": 0.3678935170173645,  "confidence": 0.628430187702179},
    {"id": 8,  "name": "right_elbow",    "x": 0.4093763530254364,  "y": 0.3831363916397095,  "confidence": 0.9928102493286133},
    {"id": 9,  "name": "left_wrist",     "x": 0.0,                 "y": 0.0,                 "confidence": 0.47565630078315735},
    {"id": 10, "name": "right_wrist",    "x": 0.44086283445358276, "y": 0.5217946171760559,  "confidence": 0.9743650555610657},
    {"id": 11, "name": "left_hip",       "x": 0.42167890071868896, "y": 0.4936578869819641,  "confidence": 0.9920280575752258},
    {"id": 12, "name": "right_hip",      "x": 0.3994448781013489,  "y": 0.49802303314208984, "confidence": 0.9982593655586243},
    {"id": 13, "name": "left_knee",      "x": 0.47539588809013367, "y": 0.6768141388893127,  "confidence": 0.9848501682281494},
    {"id": 14, "name": "right_knee",     "x": 0.36448314785957336, "y": 0.6863085627555847,  "confidence": 0.9961143732070923},
    {"id": 15, "name": "left_ankle",     "x": 0.5105754137039185,  "y": 0.8785256743431091,  "confidence": 0.9521668553352356},
    {"id": 16, "name": "right_ankle",    "x": 0.2795264422893524,  "y": 0.8489685654640198,  "confidence": 0.976803183555603},
]

KEYPOINTS_PER_FRAME = [
    {"frame": 1, "time": 0.03, "yolo_keypoints": KP01},
    {"frame": 2, "time": 0.06, "yolo_keypoints": KP01},
    {"frame": 3, "time": 0.10, "yolo_keypoints": KP01},
]

STANDARD_ANGLE = {
    "left_elbow": 160, "right_elbow": 160,
    "left_shoulder": 40, "right_shoulder": 40,
    "left_knee": 90, "right_knee": 90,
    "left_hip": 100, "right_hip": 100
}

ANGLE_SEQUENCE = [
    {"left_elbow": 169.1, "right_elbow": 162.7, "left_shoulder": 0.9,  "right_shoulder": 2.1,  "left_knee": 173.4, "right_knee": 162.2, "left_hip": 163.8, "right_hip": 173.9},
    {"left_elbow": 169.3, "right_elbow": 163.4, "left_shoulder": 0.5,  "right_shoulder": 1.6,  "left_knee": 150.0, "right_knee": 148.0, "left_hip": 140.0, "right_hip": 141.0},
    {"left_elbow": 168.9, "right_elbow": 162.1, "left_shoulder": 1.1,  "right_shoulder": 2.5,  "left_knee": 172.9, "right_knee": 161.8, "left_hip": 163.2, "right_hip": 174.1},
]

class TestAnalyzeMotion:

    def test_returns_correct_structure_without_reference(self):
        result = analyze_motion(
            KEYPOINTS_PER_FRAME,
            STANDARD_ANGLE,
            reference_angle_sequence=None
        )

        assert "accuracy_score"           in result
        assert "risk_scores"              in result
        assert "highest_risk_frame_index" in result
        assert "angles_per_frame"         in result
        assert "exercise_match"           in result

        assert isinstance(result["accuracy_score"],           float)
        assert isinstance(result["risk_scores"],              tuple)
        assert isinstance(result["highest_risk_frame_index"], int)
        assert isinstance(result["angles_per_frame"],         list)
        assert result["exercise_match"] is None

        for item in result["angles_per_frame"]:
            assert "frame"      in item
            assert "time"       in item
            assert "angles"     in item
            assert "risk_score" in item

    def test_returns_exercise_match_when_reference_provided(self):
        result = analyze_motion(
            KEYPOINTS_PER_FRAME,
            STANDARD_ANGLE,
            reference_angle_sequence=ANGLE_SEQUENCE,
            exercise_name="Squat_men"
        )

        assert result["exercise_match"] is not None

        match = result["exercise_match"]
        assert "similarity_score"     in match
        assert "average_angle_error"  in match
        assert "movement_range_error" in match
        assert "phase_shift"          in match
        assert "compared_joints"      in match
        assert "rule_check"           in match

    def test_empty_keypoints_raises_invalid_keypoints_exception(self):
        with pytest.raises(InvalidKeypointsException) as exc_info:
            analyze_motion(
                [],
                STANDARD_ANGLE,
                reference_angle_sequence=None
            )

        assert "No body joints could be reliably identified" in str(exc_info.value)