import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call

from PIL import Image

from app.services.graph_service import save_highest_risk_frame


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


def _make_cap(success: bool = True):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cap = MagicMock()
    cap.read.return_value = (success, frame if success else None)
    return cap


class TestSaveHighestRiskFrame:

    @patch("app.services.graph_service.cv2.imwrite")
    @patch("app.services.graph_service.cv2.VideoCapture")
    def test_returns_output_path_without_keypoints(
        self, mock_cap_cls, mock_imwrite
    ):
        mock_cap_cls.return_value = _make_cap(success=True)

        result = save_highest_risk_frame("exercise_vid.mp4", 150, None)

        assert isinstance(result, Image.Image)

    @patch("app.services.graph_service.draw_skeleton_on_frame")
    @patch("app.services.graph_service.cv2.imwrite")
    @patch("app.services.graph_service.cv2.VideoCapture")
    def test_draws_skeleton_when_keypoints_provided(
        self, mock_cap_cls, mock_imwrite, mock_draw
    ):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap_cls.return_value = _make_cap(success=True)
        mock_draw.return_value = frame

        result = save_highest_risk_frame("exercise_vid.mp4", 150, KP01)

        mock_draw.assert_called_once()
        assert isinstance(result, Image.Image)

    @patch("app.services.graph_service.cv2.VideoCapture")
    def test_raises_exception_when_frame_cannot_be_extracted(
        self, mock_cap_cls
    ):
        mock_cap_cls.return_value = _make_cap(success=False)

        with pytest.raises(Exception) as exc_info:
            save_highest_risk_frame("exercise_vid.mp4", 99999, None)

        assert "Failed to extract frame." in str(exc_info.value)
