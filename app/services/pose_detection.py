import cv2
import numpy as np

from app.exceptions import (
    KeypointNotDetectedException,
    pose_module_notFound_error
)

MODEL_PATH = "assets/yolo11m-pose.pt"

CONF_THRES = 0.3
KP_CONF_THRES = 0.35
MIN_FRAME_SHARPNESS = 30.0
_POSE_MODEL = None

REQUIRED_KEYPOINT_NAMES = {
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
}

KP_NAME = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle"
}

def _load_pose_dependencies():
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise pose_module_notFound_error() from error
    return YOLO

def _get_pose_model():
    global _POSE_MODEL

    if _POSE_MODEL is None:
        yolo_model = _load_pose_dependencies()
        _POSE_MODEL = yolo_model(MODEL_PATH)

    return _POSE_MODEL


def _update_progress(progress_callback, percent, message):
    if progress_callback is None:
        return

    progress_callback({
        "step": "detecting",
        "message": message,
        "percent": percent
    })


def _open_video_capture(file: str):
    cap = cv2.VideoCapture(file)
    orientation_auto = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    if orientation_auto is not None:
        cap.set(orientation_auto, 1)
    return cap


def _has_reliable_keypoints(conf):
    if len(conf) == 0 or np.mean(conf) < KP_CONF_THRES:
        return False

    required_confidences = [
        conf[index]
        for index, name in KP_NAME.items()
        if name in REQUIRED_KEYPOINT_NAMES and index < len(conf)
    ]

    if len(required_confidences) < len(REQUIRED_KEYPOINT_NAMES):
        return False

    return min(required_confidences) >= KP_CONF_THRES


def _is_frame_sharp(frame):
    if frame is None or frame.size == 0:
        return False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    return sharpness >= MIN_FRAME_SHARPNESS


# Detect body keypoints from uploaded exercise video
def detect_body_keypoints(file: str, progress_callback=None):
    model = _get_pose_model()

    cap = _open_video_capture(file)
    fps = (
            cap.get(cv2.CAP_PROP_FPS)
            or 30
    )
    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
        or 0
    )
    keypoints_per_frame = []
    FRAME_SKIP = 2 
    frame_id = 0
    last_reported_percent = 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        if frame_id % FRAME_SKIP != 0:
            continue

        if not _is_frame_sharp(frame):
            continue

        if total_frames > 0:
            percent = 30 + int(
                (frame_id / total_frames)
                * 30
            )
            if percent >= last_reported_percent + 5:
                last_reported_percent = percent
                _update_progress(
                    progress_callback,
                    min(percent, 60),
                    "Detecting pose..."
                )

        results = model(
            frame,
            conf=CONF_THRES,
            verbose=False
        )
        detected = False

        for result in results:
            if result.keypoints is None:
                continue
            if result.keypoints.xyn is None:
                continue
            if len(result.keypoints.xyn) == 0:
                continue
            xyn = (
                result
                .keypoints
                .xyn[0]
                .cpu()
                .numpy()
            )
            if result.keypoints.conf is not None:
                conf = (
                    result
                    .keypoints
                    .conf[0]
                    .cpu()
                    .numpy()
                )
            else:
                conf = np.ones(len(xyn))
            if not _has_reliable_keypoints(conf):
                continue
            frame_data = {
                "frame": frame_id,
                "time": round(
                    frame_id / fps,
                    2
                ),
                "yolo_keypoints": [
                    {
                        "id": i,
                        "name": KP_NAME.get(
                            i,
                            "unknown"
                        ),
                        "x": float(xyn[i][0]),
                        "y": float(xyn[i][1]),
                        "confidence": float(conf[i])
                    }
                    for i in range(len(xyn))
                ]
            }
            keypoints_per_frame.append(
                frame_data
            )
            detected = True
            break

        if not detected:
            continue

    cap.release()

    _update_progress(
        progress_callback,
        60,
        "Pose detection completed."
    )

    # Validation
    if len(keypoints_per_frame) == 0:
        raise KeypointNotDetectedException()

    return keypoints_per_frame

EARLY_CHECK_FRAMES = 20  # detect แค่ 20 frames แรกเพื่อ validate

def detect_sample_keypoints(
        file: str,
        sample_count: int = EARLY_CHECK_FRAMES
):
    """Detect keypoints from sampled frames for exercise validation."""

    model = _get_pose_model()

    cap = _open_video_capture(file)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    )

    keypoints_per_frame = []

    frame_id = 0

    step = max(
        1,
        total_frames // sample_count
    )

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_id += 1

        if frame_id % step != 0:
            continue

        if not _is_frame_sharp(frame):
            continue

        results = model(
            frame,
            conf=CONF_THRES,
            verbose=False
        )

        for result in results:

            if result.keypoints is None:
                continue

            if result.keypoints.xyn is None:
                continue

            if len(result.keypoints.xyn) == 0:
                continue

            xyn = (
                result.keypoints.xyn[0]
                .cpu()
                .numpy()
            )

            conf = (
                result.keypoints.conf[0]
                .cpu()
                .numpy()
                if result.keypoints.conf is not None
                else np.ones(len(xyn))
            )

            if not _has_reliable_keypoints(conf):
                continue

            keypoints_per_frame.append({
                "frame": frame_id,
                "time": round(
                    frame_id / fps,
                    2
                ),
                "yolo_keypoints": [
                    {
                        "id": i,
                        "name": KP_NAME.get(
                            i,
                            "unknown"
                        ),
                        "x": float(xyn[i][0]),
                        "y": float(xyn[i][1]),
                        "confidence": float(conf[i])
                    }
                    for i in range(len(xyn))
                ]
            })

            break

        if len(keypoints_per_frame) >= sample_count:
            break

    cap.release()

    if not keypoints_per_frame:
        raise KeypointNotDetectedException()

    return keypoints_per_frame
