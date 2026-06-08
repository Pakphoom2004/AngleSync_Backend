import cv2
import numpy as np
import os

from app.exceptions import (
    KeypointNotDetectedException,
    ServiceException
)

MODEL_PATH = "assets/yolo11m-pose.pt"

CONF_THRES = 0.3
KP_CONF_THRES = 0.15
_POSE_MODEL = None

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
        raise ServiceException(
            "Missing dependency: ultralytics. Install it before running analysis."
        ) from error

    return YOLO


def _get_pose_model():
    global _POSE_MODEL

    if _POSE_MODEL is None:
        yolo_model = _load_pose_dependencies()
        _POSE_MODEL = yolo_model(MODEL_PATH)

    return _POSE_MODEL


def _report_progress(progress_callback, percent, message):
    if progress_callback is None:
        return

    progress_callback({
        "step": "detecting",
        "message": message,
        "percent": percent
    })


# Detect body keypoints from uploaded exercise video
def detect_body_keypoints(file: str, progress_callback=None):
    model = _get_pose_model()

    cap = cv2.VideoCapture(file)
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

        if total_frames > 0:
            percent = 30 + int(
                (frame_id / total_frames)
                * 30
            )
            if percent >= last_reported_percent + 5:
                last_reported_percent = percent
                _report_progress(
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
            if np.mean(conf) < KP_CONF_THRES:
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

    _report_progress(
        progress_callback,
        60,
        "Pose detection completed."
    )

    # Validation
    if len(keypoints_per_frame) == 0:
        raise KeypointNotDetectedException()

    return keypoints_per_frame

EARLY_CHECK_FRAMES = 20  # detect แค่ 20 frames แรกเพื่อ validate

def detect_sample_keypoints(file: str, sample_count: int = EARLY_CHECK_FRAMES):
    """Detect keypoints จาก sample frames เพื่อ validate exercise เท่านั้น"""
    model = _get_pose_model()
    cap = cv2.VideoCapture(file)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    keypoints_per_frame = []
    frame_id = 0
    step = max(1, total_frames // sample_count)  # กระจาย sample ตลอด video

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        if frame_id % step != 0:
            continue

        results = model(frame, conf=CONF_THRES, verbose=False)

        for result in results:
            if result.keypoints is None:
                continue
            if result.keypoints.xyn is None:
                continue
            if len(result.keypoints.xyn) == 0:
                continue

            xyn = result.keypoints.xyn[0].cpu().numpy()
            conf = (
                result.keypoints.conf[0].cpu().numpy()
                if result.keypoints.conf is not None
                else np.ones(len(xyn))
            )

            if np.mean(conf) < KP_CONF_THRES:
                continue

            keypoints_per_frame.append({
                "frame": frame_id,
                "time": round(frame_id / fps, 2),
                "yolo_keypoints": [
                    {
                        "id": i,
                        "name": KP_NAME.get(i, "unknown"),
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
    return keypoints_per_frame

ALLOWED_EXTENSIONS = {".mp4", ".mov"}
MAX_DURATION_SECONDS = 60

def verify_file_type(file: str):
    ext = os.path.splitext(file)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ServiceException(
            "Invalid file format. Please upload an MP4 or MOV file."
        )

def verify_video_duration(file: str):
    cap = cv2.VideoCapture(file)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    duration = total_frames / fps
    if duration > MAX_DURATION_SECONDS:
        raise ServiceException(
            "Video exceeds 60 seconds."
        )