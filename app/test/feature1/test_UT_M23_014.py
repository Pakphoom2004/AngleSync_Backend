import pytest
from unittest.mock import MagicMock, patch

from app.services.analysis_results import process_video_analysis
from app.exceptions import ServiceException, ExerciseMismatchException


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

REFERENCE_DATA = {
    "exercise_name": "Squat_men",
    "average_angles": {
        "left_elbow": 169.1, "right_elbow": 162.7,
        "left_shoulder": 0.9, "right_shoulder": 2.1,
        "left_knee": 173.4, "right_knee": 162.2,
        "left_hip": 163.8, "right_hip": 173.9
    },
    "angle_sequence": [
        {"left_elbow": 169.1, "right_elbow": 162.7, "left_shoulder": 0.9,  "right_shoulder": 2.1,  "left_knee": 173.4, "right_knee": 162.2, "left_hip": 163.8, "right_hip": 173.9},
        {"left_elbow": 169.3, "right_elbow": 163.4, "left_shoulder": 0.5,  "right_shoulder": 1.6,  "left_knee": 150.0, "right_knee": 148.0, "left_hip": 140.0, "right_hip": 141.0},
        {"left_elbow": 168.9, "right_elbow": 162.1, "left_shoulder": 1.1,  "right_shoulder": 2.5,  "left_knee": 172.9, "right_knee": 161.8, "left_hip": 163.2, "right_hip": 174.1},
    ]
}

MOCK_ANALYSIS_RESULT = {
    "accuracy_score": 85.4,
    "risk_scores": (10.2, 47.76, 85.4),
    "highest_risk_frame_index": 2,
    "angles_per_frame": [
        {"frame": 1, "time": 0.03, "angles": {}, "keypoints": KP01, "risk_score": 10.2},
        {"frame": 2, "time": 0.06, "angles": {}, "keypoints": KP01, "risk_score": 47.76},
        {"frame": 3, "time": 0.10, "angles": {}, "keypoints": KP01, "risk_score": 85.4},
    ],
    "exercise_match": {
        "similarity_score": 95.0,
        "average_angle_error": 2.1,
        "movement_range_error": 1.5,
        "phase_shift": 0,
        "compared_joints": ["left_knee", "right_knee"],
        "rule_check": {"passed": True}
    }
}

MOCK_FEEDBACK = {
    "prompt": "You are an expert fitness coach.",
    "feedback": {
        "form_summary":    "Good squat depth.",
        "injury_risk":     "Low risk.",
        "corrective_cues": "Keep knees over toes.",
        "practice_plan":   "Practice 3 sets daily."
    }
}


def _patch_all(
    mock_verify_type=None,
    mock_verify_duration=None,
    mock_get_reference=None,
    mock_detect_sample=None,
    mock_validate=None,
    mock_detect_body=None,
    mock_analyze=None,
    mock_graph=None,
    mock_save_frame=None,
    mock_feedback=None,
):
    patches = {
        "app.services.analysis_results.verify_file_type":           mock_verify_type     or MagicMock(),
        "app.services.analysis_results.verify_video_duration":      mock_verify_duration or MagicMock(return_value=None),
        "app.services.analysis_results.get_reference_angles_from_db": mock_get_reference or MagicMock(return_value=REFERENCE_DATA),
        "app.services.analysis_results.detect_sample_keypoints":    mock_detect_sample   or MagicMock(return_value=KEYPOINTS_PER_FRAME),
        "app.services.analysis_results.validate_exercise_match":    mock_validate        or MagicMock(),
        "app.services.analysis_results.detect_body_keypoints":      mock_detect_body     or MagicMock(return_value=KEYPOINTS_PER_FRAME),
        "app.services.analysis_results.analyze_motion":             mock_analyze         or MagicMock(return_value=MOCK_ANALYSIS_RESULT),
        "app.services.analysis_results.generate_risk_graph":        mock_graph           or MagicMock(return_value=MagicMock()),
        "app.services.analysis_results.save_highest_risk_frame":    mock_save_frame      or MagicMock(return_value="data/outputs/highest_risk_frame.jpg"),
        "app.services.analysis_results.generate_advanced_feedback": mock_feedback        or MagicMock(return_value=MOCK_FEEDBACK),
        "app.services.analysis_results.extract_joint_angles":       MagicMock(return_value={"left_knee": 173.4}),
    }
    return patches


class TestProcessVideoAnalysis:

    def test_returns_completed_status_when_video_matches(self):
        with patch.multiple("", **_patch_all()):
            with patch("app.services.analysis_results.verify_file_type"),\
                 patch("app.services.analysis_results.verify_video_duration", return_value=None),\
                 patch("app.services.analysis_results.get_reference_angles_from_db", return_value=REFERENCE_DATA),\
                 patch("app.services.analysis_results.detect_sample_keypoints", return_value=KEYPOINTS_PER_FRAME),\
                 patch("app.services.analysis_results.extract_joint_angles", return_value={"left_knee": 173.4}),\
                 patch("app.services.analysis_results.validate_exercise_match"),\
                 patch("app.services.analysis_results.detect_body_keypoints", return_value=KEYPOINTS_PER_FRAME),\
                 patch("app.services.analysis_results.analyze_motion", return_value=MOCK_ANALYSIS_RESULT),\
                 patch("app.services.analysis_results.generate_risk_graph", return_value=MagicMock()),\
                 patch("app.services.analysis_results.save_highest_risk_frame", return_value="data/outputs/highest_risk_frame.jpg"),\
                 patch("app.services.analysis_results.generate_advanced_feedback", return_value=MOCK_FEEDBACK):

                result = process_video_analysis("exercise_vid.mp4", 4, None)

        assert result["status"]      == "completed"
        assert result["can_analyze"] is True
        assert isinstance(result["score"], float)
        assert result["score_scale"] == 100
        assert result["risk_level"]  in ("GOOD", "NORMAL", "DANGEROUS")
        assert "exercise_match"  in result
        assert "selected_frame"  in result
        assert "graph_data"      in result
        assert "feedback"        in result

    def test_returns_mismatch_status_when_exercise_does_not_match(self):
        mismatch_error = ExerciseMismatchException(
            similarity_score=20.0,
            details={
                "average_angle_error":  80.0,
                "movement_range_error": 60.0,
                "rule_check": {"passed": False}
            }
        )

        with patch("app.services.analysis_results.verify_file_type"),\
             patch("app.services.analysis_results.verify_video_duration", return_value=None),\
             patch("app.services.analysis_results.get_reference_angles_from_db", return_value=REFERENCE_DATA),\
             patch("app.services.analysis_results.detect_sample_keypoints", return_value=KEYPOINTS_PER_FRAME),\
             patch("app.services.analysis_results.extract_joint_angles", return_value={"left_knee": 90.0}),\
             patch("app.services.analysis_results.validate_exercise_match", side_effect=mismatch_error):

            result = process_video_analysis("wrong_exercise.mp4", 4, None)

        assert result["status"]                        == "exercise_mismatch"
        assert result["can_analyze"]                   is False
        assert result["score"]                         == 0.0
        assert result["risk_level"]                    == "MISMATCH"
        assert result["exercise_match"]["is_match"]    is False

    def test_reports_progress_via_callback(self):
        mock_callback = MagicMock()

        with patch("app.services.analysis_results.verify_file_type"),\
             patch("app.services.analysis_results.verify_video_duration", return_value=None),\
             patch("app.services.analysis_results.get_reference_angles_from_db", return_value=REFERENCE_DATA),\
             patch("app.services.analysis_results.detect_sample_keypoints", return_value=KEYPOINTS_PER_FRAME),\
             patch("app.services.analysis_results.extract_joint_angles", return_value={"left_knee": 173.4}),\
             patch("app.services.analysis_results.validate_exercise_match"),\
             patch("app.services.analysis_results.detect_body_keypoints", return_value=KEYPOINTS_PER_FRAME),\
             patch("app.services.analysis_results.analyze_motion", return_value=MOCK_ANALYSIS_RESULT),\
             patch("app.services.analysis_results.generate_risk_graph", return_value=MagicMock()),\
             patch("app.services.analysis_results.save_highest_risk_frame", return_value="data/outputs/highest_risk_frame.jpg"),\
             patch("app.services.analysis_results.generate_advanced_feedback", return_value=MOCK_FEEDBACK):

            process_video_analysis("exercise_vid.mp4", 4, mock_callback)

        steps = [
            call[0][0]["step"]
            for call in mock_callback.call_args_list
        ]
        assert "validating"          in steps
        assert "detecting"           in steps
        assert "analyzing"           in steps
        assert "generating_graph"    in steps
        assert "generating_feedback" in steps

    def test_returns_fallback_feedback_when_ai_fails(self):
        with patch("app.services.analysis_results.verify_file_type"),\
             patch("app.services.analysis_results.verify_video_duration", return_value=None),\
             patch("app.services.analysis_results.get_reference_angles_from_db", return_value=REFERENCE_DATA),\
             patch("app.services.analysis_results.detect_sample_keypoints", return_value=KEYPOINTS_PER_FRAME),\
             patch("app.services.analysis_results.extract_joint_angles", return_value={"left_knee": 173.4}),\
             patch("app.services.analysis_results.validate_exercise_match"),\
             patch("app.services.analysis_results.detect_body_keypoints", return_value=KEYPOINTS_PER_FRAME),\
             patch("app.services.analysis_results.analyze_motion", return_value=MOCK_ANALYSIS_RESULT),\
             patch("app.services.analysis_results.generate_risk_graph", return_value=MagicMock()),\
             patch("app.services.analysis_results.save_highest_risk_frame", return_value="data/outputs/highest_risk_frame.jpg"),\
             patch("app.services.analysis_results.generate_advanced_feedback",
                   side_effect=ServiceException("AI service failed.")):

            result = process_video_analysis("exercise_vid.mp4", 4, None)

        assert result["status"]                    == "completed"
        assert result["feedback"]["form_summary"]  == "AI feedback unavailable."