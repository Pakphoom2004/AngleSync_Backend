import pytest
import numpy as np

from app.services.graph_service import draw_skeleton_on_frame
from unittest.mock import patch, call

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

KP01_ZERO_LEFT_EYE = [
    kp if kp["name"] != "left_eye"
    else {**kp, "confidence": 0.0}
    for kp in KP01
]
KP01_ZERO_LEFT_SHOULDER = [
    kp if kp["name"] != "left_shoulder"
    else {**kp, "confidence": 0.0}
    for kp in KP01
]

class TestDrawSkeletonOnFrame:

    def test_returns_same_shape_as_input_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = draw_skeleton_on_frame(frame, KP01)

        assert result.shape == (480, 640, 3)

    def test_skips_keypoint_with_zero_confidence(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("app.services.graph_service.cv2.circle") as mock_circle, \
                patch("app.services.graph_service.cv2.line"):
            draw_skeleton_on_frame(frame, KP01_ZERO_LEFT_SHOULDER)

            # KP01 มี 17 จุด ถ้า skip left_shoulder → วาดได้แค่ 16 จุด
            # cv2.circle ถูกเรียก 2 ครั้งต่อ 1 จุด (circle ใหญ่ + เล็ก)
            call_count = mock_circle.call_count
            assert call_count == 16 * 2, \
                f"Expected 32 circle calls (16 joints × 2), got {call_count}"

    def test_draws_skeleton_lines_with_correct_color(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = draw_skeleton_on_frame(frame, KP01)

        # ตรวจว่ามี pixel ที่มีสี (0, 210, 90) อยู่ในภาพ
        skeleton_color = [0, 210, 90]
        has_skeleton_color = np.any(
            np.all(result == skeleton_color, axis=2)
        )
        assert has_skeleton_color, "No skeleton line color (0, 210, 90) found in frame"

        # ตรวจว่ามี pixel ที่มีสี (0, 0, 255) อยู่ในภาพ
        joint_color = [0, 0, 255]
        has_joint_color = np.any(
            np.all(result == joint_color, axis=2)
        )
        assert has_joint_color, "No joint circle color (0, 0, 255) found in frame"