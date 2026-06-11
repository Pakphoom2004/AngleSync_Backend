import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.services.pose_detection import detect_body_keypoints
from app.exceptions import KeypointNotDetectedException


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


def _make_yolo_result():
    xyn_data  = np.array([[kp["x"], kp["y"]]  for kp in KP01], dtype=np.float32)
    conf_data = np.array([kp["confidence"]     for kp in KP01], dtype=np.float32)

    xyn_tensor = MagicMock()
    xyn_tensor.cpu.return_value.numpy.return_value = xyn_data

    conf_tensor = MagicMock()
    conf_tensor.cpu.return_value.numpy.return_value = conf_data

    result = MagicMock()
    result.keypoints = MagicMock()
    result.keypoints.xyn.__len__      = MagicMock(return_value=1)
    result.keypoints.xyn.__getitem__  = MagicMock(return_value=xyn_tensor)
    result.keypoints.conf.__getitem__ = MagicMock(return_value=conf_tensor)

    return result


def _make_cap(frame_count: int = 6, fps: float = 30.0):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    reads = [(True, frame)] * frame_count + [(False, None)]

    cap = MagicMock()
    cap.get.side_effect = lambda prop: fps if prop == 5 else float(frame_count)
    cap.read.side_effect = reads
    return cap


class TestDetectBodyKeypoints:

    @patch("app.services.pose_detection._get_pose_model")
    @patch("app.services.pose_detection.cv2.VideoCapture")
    def test_returns_keypoints_and_calls_callback(
        self, mock_cap_cls, mock_get_model
    ):
        mock_cap_cls.return_value = _make_cap(frame_count=6)
        mock_get_model.return_value = MagicMock(
            return_value=[_make_yolo_result()]
        )
        mock_callback = MagicMock()

        result = detect_body_keypoints("exercise_vid.mp4", mock_callback)

        assert len(result) > 0
        assert "yolo_keypoints" in result[0]

        for call in mock_callback.call_args_list:
            data = call[0][0]
            if data.get("message") == "Detecting pose...":
                assert 30 <= data["percent"] <= 60

    @patch("app.services.pose_detection._get_pose_model")
    @patch("app.services.pose_detection.cv2.VideoCapture")
    def test_skips_odd_frames_according_to_frame_skip(
        self, mock_cap_cls, mock_get_model
    ):
        mock_cap_cls.return_value = _make_cap(frame_count=6)
        mock_get_model.return_value = MagicMock(
            return_value=[_make_yolo_result()]
        )

        result = detect_body_keypoints("exercise_vid.mp4", None)

        frame_ids = [r["frame"] for r in result]
        for fid in frame_ids:
            assert fid % 2 == 0, f"frame {fid} is odd, should be skipped"

    @patch("app.services.pose_detection._get_pose_model")
    @patch("app.services.pose_detection.cv2.VideoCapture")
    def test_raises_exception_when_no_person_detected(
        self, mock_cap_cls, mock_get_model
    ):
        mock_cap_cls.return_value = _make_cap(frame_count=3)
        empty_result = MagicMock()
        empty_result.keypoints = None
        mock_get_model.return_value = MagicMock(return_value=[empty_result])

        with pytest.raises(KeypointNotDetectedException) as exc_info:
            detect_body_keypoints("frame_ai.mp4", None)

        assert "Detection failed" in str(exc_info.value) or \
               "visible" in str(exc_info.value)

    @patch("app.services.pose_detection._get_pose_model")
    @patch("app.services.pose_detection.cv2.VideoCapture")
    def test_completes_without_error_when_callback_is_none(
        self, mock_cap_cls, mock_get_model
    ):
        mock_cap_cls.return_value = _make_cap(frame_count=6)
        mock_get_model.return_value = MagicMock(
            return_value=[_make_yolo_result()]
        )

        result = detect_body_keypoints("exercise_vid.mp4", None)

        assert len(result) > 0