import pytest
from unittest.mock import MagicMock, patch

from app.services.analysis_results import process_video_analysis
from app.exceptions import ServiceException, ExerciseMismatchException


KP01 = [
    {"id": 0, "name": "nose", "x": 0.4680759608745575, "y": 0.13428252935409546, "confidence": 0.8291665315628052},
    {"id": 1, "name": "left_eye", "x": 0.0, "y": 0.0, "confidence": 0.20284414291381836},
    {"id": 2, "name": "right_eye", "x": 0.4652771055698395, "y": 0.11990076303482056, "confidence": 0.8898438811302185},
    {"id": 3, "name": "left_ear", "x": 0.0, "y": 0.0, "confidence": 0.050865743309259415},
    {"id": 4, "name": "right_ear", "x": 0.43922296166419983, "y": 0.11978987604379654, "confidence": 0.9512590169906616},
    {"id": 5, "name": "left_shoulder", "x": 0.42056581377983093, "y": 0.2169395238161087, "confidence": 0.9129444360733032},
    {"id": 6, "name": "right_shoulder", "x": 0.4168631434440613, "y": 0.22159205377101898, "confidence": 0.9958706498146057},
    {"id": 7, "name": "left_elbow", "x": 0.41852816939353943, "y": 0.3678935170173645, "confidence": 0.628430187702179},
    {"id": 8, "name": "right_elbow", "x": 0.4093763530254364, "y": 0.3831363916397095, "confidence": 0.9928102493286133},
    {"id": 9, "name": "left_wrist", "x": 0.0, "y": 0.0, "confidence": 0.47565630078315735},
    {"id": 10, "name": "right_wrist", "x": 0.44086283445358276, "y": 0.5217946171760559, "confidence": 0.9743650555610657},
    {"id": 11, "name": "left_hip", "x": 0.42167890071868896, "y": 0.4936578869819641, "confidence": 0.9920280575752258},
    {"id": 12, "name": "right_hip", "x": 0.3994448781013489, "y": 0.49802303314208984, "confidence": 0.9982593655586243},
    {"id": 13, "name": "left_knee", "x": 0.47539588809013367, "y": 0.6768141388893127, "confidence": 0.9848501682281494},
    {"id": 14, "name": "right_knee", "x": 0.36448314785957336, "y": 0.6863085627555847, "confidence": 0.9961143732070923},
    {"id": 15, "name": "left_ankle", "x": 0.5105754137039185, "y": 0.8785256743431091, "confidence": 0.9521668553352356},
    {"id": 16, "name": "right_ankle", "x": 0.2795264422893524, "y": 0.8489685654640198, "confidence": 0.976803183555603},
]


KEYPOINTS_PER_FRAME = [
    {"frame": 1, "time": 0.03, "yolo_keypoints": KP01},
    {"frame": 2, "time": 0.06, "yolo_keypoints": KP01},
    {"frame": 3, "time": 0.10, "yolo_keypoints": KP01},
]


REFERENCE_DATA = {
    "exercise_name": "Squat_men",
    "average_angles": {"left_elbow": 169.1, "right_elbow": 162.7, "left_shoulder": 0.9, "right_shoulder": 2.1, "left_knee": 173.4, "right_knee": 162.2, "left_hip": 163.8, "right_hip": 173.9},
    "angle_sequence": [
        {"left_elbow": 169.1, "right_elbow": 162.7, "left_shoulder": 0.9, "right_shoulder": 2.1, "left_knee": 173.4, "right_knee": 162.2, "left_hip": 163.8, "right_hip": 173.9},
        {"left_elbow": 169.3, "right_elbow": 163.4, "left_shoulder": 0.5, "right_shoulder": 1.6, "left_knee": 150.0, "right_knee": 148.0, "left_hip": 140.0, "right_hip": 141.0},
        {"left_elbow": 168.9, "right_elbow": 162.1, "left_shoulder": 1.1, "right_shoulder": 2.5, "left_knee": 172.9, "right_knee": 161.8, "left_hip": 163.2, "right_hip": 174.1},
    ],
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
        "rule_check": {"passed": True},
    },
}


MOCK_FEEDBACK = {
    "prompt": "You are an expert fitness coach.",
    "feedback": {
        "form_summary": "Good squat depth.",
        "injury_risk": "Low risk.",
        "corrective_cues": "Keep knees over toes.",
        "practice_plan": "Practice 3 sets daily.",
    },
}


class TestProcessVideoAnalysis:

    # UT-M30-014
    @patch("app.services.analysis_results.get_public_url")
    @patch("app.services.analysis_results.upload_bytes")
    @patch("app.services.analysis_results.verify_file_type")
    @patch(
        "app.services.analysis_results.verify_video_duration",
        return_value=None,
    )
    @patch(
        "app.services.analysis_results.get_reference_angles_from_db",
        return_value=REFERENCE_DATA,
    )
    @patch(
        "app.services.analysis_results.detect_sample_keypoints",
        return_value=KEYPOINTS_PER_FRAME,
    )
    @patch(
        "app.services.analysis_results.extract_joint_angles",
        return_value={"left_knee": 173.4},
    )
    @patch("app.services.analysis_results.validate_exercise_match")
    @patch(
        "app.services.analysis_results.detect_body_keypoints",
        return_value=KEYPOINTS_PER_FRAME,
    )
    @patch(
        "app.services.analysis_results.analyze_motion",
        return_value=MOCK_ANALYSIS_RESULT,
    )
    @patch(
        "app.services.analysis_results.generate_risk_graph",
        return_value=MagicMock(),
    )
    @patch(
        "app.services.analysis_results.save_highest_risk_frame",
    )
    @patch(
        "app.services.analysis_results.generate_advanced_feedback",
        return_value=MOCK_FEEDBACK,
    )
    def test_uses_supabase_public_url_for_highest_risk_image(
        self,
        mock_feedback,
        mock_save_frame,
        mock_graph,
        mock_analyze,
        mock_detect_body,
        mock_validate,
        mock_extract_angles,
        mock_detect_sample,
        mock_reference,
        mock_duration,
        mock_verify_type,
        mock_upload_bytes,
        mock_get_public_url,
    ):
        # save_highest_risk_frame returns a PIL-like object
        frame_image = MagicMock()
        frame_image.save.side_effect = (
            lambda buffer, format: buffer.write(b"fake-image-bytes")
        )

        mock_save_frame.return_value = frame_image

        mock_get_public_url.return_value = (
            "https://test.garage.example.com/my-bucket/"
            "highest_risk_frame_abc.jpg"
        )

        with patch(
            "app.services.analysis_results.uuid.uuid4"
        ) as mock_uuid:
            mock_uuid.return_value.hex = "abc"

            result = process_video_analysis(
                "exercise_vid.mp4",
                4,
                None,
            )

        expected_url = (
            "https://test.garage.example.com/my-bucket/"
            "highest_risk_frame_abc.jpg"
        )

        assert result["selected_frame"]["image"] == expected_url

        assert (
            result["graph_data"]["highest_risk_image_url"]
            == expected_url
        )

        mock_upload_bytes.assert_called_once()

        mock_get_public_url.assert_called_once_with(
            "highest_risk_frame_abc.jpg"
        )
