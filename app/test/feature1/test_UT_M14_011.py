import pytest

from app.services.motion_analysis import extract_joint_angles


FRAME01 = {
    "frame": 3,
    "time": 0.1,
    "yolo_keypoints": [
        {"name": "left_shoulder",  "x": 0.42056581377983093, "y": 0.2169395238161087},
        {"name": "right_shoulder", "x": 0.4168631434440613,  "y": 0.22159205377101898},
        {"name": "left_elbow",     "x": 0.41852816939353943, "y": 0.3678935170173645},
        {"name": "right_elbow",    "x": 0.4093763530254364,  "y": 0.3831363916397095},
        {"name": "left_wrist",     "x": 0.44086283445358276, "y": 0.5217946171760559},
        {"name": "right_wrist",    "x": 0.44086283445358276, "y": 0.5217946171760559},
        {"name": "left_hip",       "x": 0.42167890071868896, "y": 0.4936578869819641},
        {"name": "right_hip",      "x": 0.3994448781013489,  "y": 0.49802303314208984},
        {"name": "left_knee",      "x": 0.47539588809013367, "y": 0.6768141388893127},
        {"name": "right_knee",     "x": 0.36448314785957336, "y": 0.6863085627555847},
        {"name": "left_ankle",     "x": 0.5105754137039185,  "y": 0.8785256743431091},
        {"name": "right_ankle",    "x": 0.2795264422893524,  "y": 0.8489685654640198},
    ]
}

FRAME02 = {
    "frame": 3,
    "time": 0.1,
    "yolo_keypoints": [
        {"name": "left_shoulder",  "x": 0.42056581377983093, "y": 0.2169395238161087},
        {"name": "right_shoulder", "x": 0.4168631434440613,  "y": 0.22159205377101898},
        {"name": "left_elbow",     "x": 0.41852816939353943, "y": 0.3678935170173645},
        {"name": "right_elbow",    "x": 0.4093763530254364,  "y": 0.3831363916397095},
        {"name": "left_wrist",     "x": 0.44086283445358276, "y": 0.5217946171760559},
        {"name": "right_wrist",    "x": 0.44086283445358276, "y": 0.5217946171760559},
        {"name": "left_hip",       "x": 0.42167890071868896, "y": 0.4936578869819641},
        {"name": "right_hip",      "x": 0.3994448781013489,  "y": 0.49802303314208984},
        {"name": "right_knee",     "x": 0.36448314785957336, "y": 0.6863085627555847},
        {"name": "left_ankle",     "x": 0.5105754137039185,  "y": 0.8785256743431091},
        {"name": "right_ankle",    "x": 0.2795264422893524,  "y": 0.8489685654640198},
    ]
}

EXPECTED_KEYS = {
    "left_elbow", "right_elbow",
    "left_shoulder", "right_shoulder",
    "left_knee", "right_knee",
    "left_hip", "right_hip",
}


class TestExtractJointAngles:

    def test_all_keypoints_present_returns_8_angle_keys(self):
        result = extract_joint_angles(FRAME01)

        assert result is not None
        assert set(result.keys()) == EXPECTED_KEYS

    def test_missing_keypoint_returns_none(self):
        result = extract_joint_angles(FRAME02)

        assert result is None