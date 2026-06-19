import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.exceptions.keypoint_not_detected_exception import KeypointNotDetectedException
from app.services.pose_detection import detect_sample_keypoints


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


def _make_yolo_result(confidence_override=None):
    xyn_data  = np.array([[kp["x"], kp["y"]]  for kp in KP01], dtype=np.float32)
    conf_data = np.array(
        [
            confidence_override
            if confidence_override is not None
            else kp["confidence"]
            for kp in KP01
        ],
        dtype=np.float32
    )

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


def _make_cap(frame_count: int, fps: float = 30.0):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, 320:] = 255
    reads = [(True, frame)] * frame_count + [(False, None)]

    cap = MagicMock()
    cap.get.side_effect = lambda prop: fps if prop == 5 else float(frame_count)
    cap.read.side_effect = reads
    return cap


class TestDetectSampleKeypoints:

    @patch("app.services.pose_detection._get_pose_model")
    @patch("app.services.pose_detection.cv2.VideoCapture")
    def test_returns_keypoints_not_exceeding_sample_count(
        self, mock_cap_cls, mock_get_model
    ):
        mock_cap_cls.return_value = _make_cap(frame_count=30)
        mock_get_model.return_value = MagicMock(
            return_value=[_make_yolo_result()]
        )

        result = detect_sample_keypoints("exercise_vid.mp4", sample_count=5)

        assert len(result) <= 5

    @patch("app.services.pose_detection._get_pose_model")
    @patch("app.services.pose_detection.cv2.VideoCapture")
    def test_frames_are_distributed_evenly(
        self, mock_cap_cls, mock_get_model
    ):
        mock_cap_cls.return_value = _make_cap(frame_count=30)
        mock_get_model.return_value = MagicMock(
            return_value=[_make_yolo_result()]
        )

        result = detect_sample_keypoints("exercise_vid.mp4", sample_count=3)

        frame_ids = [r["frame"] for r in result]
        for i in range(1, len(frame_ids)):
            assert frame_ids[i] - frame_ids[i - 1] > 1, \
                f"frames {frame_ids[i-1]} and {frame_ids[i]} are consecutive"

    @patch("app.services.pose_detection._get_pose_model")
    @patch("app.services.pose_detection.cv2.VideoCapture")
    def test_raises_video_quality_exception_when_no_person_detected(
            self,
            mock_cap_cls,
            mock_get_model
    ):
        mock_cap_cls.return_value = _make_cap(
            frame_count=30
        )

        empty_result = MagicMock()
        empty_result.keypoints = None

        mock_get_model.return_value = MagicMock(
            return_value=[empty_result]
        )

        with pytest.raises(
                KeypointNotDetectedException
        ) as exc_info:
            detect_sample_keypoints(
                "frame_ai.mp4",
                sample_count=5
            )

        assert (
                str(exc_info.value)
                == "Keypoint not found. Please ensure that the person is visible."
        )

    @patch("app.services.pose_detection._get_pose_model")
    @patch("app.services.pose_detection.cv2.VideoCapture")
    def test_raises_keypoint_not_detected_when_keypoint_confidence_is_low(
            self,
            mock_cap_cls,
            mock_get_model
    ):
        mock_cap_cls.return_value = _make_cap(frame_count=30)
        mock_get_model.return_value = MagicMock(
            return_value=[_make_yolo_result(confidence_override=0.2)]
        )

        with pytest.raises(KeypointNotDetectedException):
            detect_sample_keypoints("blurred_video.mp4", sample_count=5)

    @patch("app.services.pose_detection._get_pose_model")
    @patch("app.services.pose_detection.cv2.VideoCapture")
    def test_raises_keypoint_not_detected_when_frames_are_blurry(
            self,
            mock_cap_cls,
            mock_get_model
    ):
        blurry_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        cap = MagicMock()
        cap.get.side_effect = lambda prop: 30.0 if prop == 5 else 30.0
        cap.read.side_effect = [(True, blurry_frame)] * 30 + [(False, None)]
        mock_cap_cls.return_value = cap
        mock_get_model.return_value = MagicMock(
            return_value=[_make_yolo_result()]
        )

        with pytest.raises(KeypointNotDetectedException):
            detect_sample_keypoints("blurred_video.mp4", sample_count=5)
