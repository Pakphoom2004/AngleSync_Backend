import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

from exceptions import KeypointNotDetectedException

MODEL_PATH = "assets/yolo11m-pose.pt"
MP_MODEL_PATH = "assets/pose_landmarker_lite.task"

CONF_THRES = 0.3
KP_CONF_THRES = 0.15

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

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


# Detect additional wrist and ankle landmarks using MediaPipe
def get_mp_points(
        frame,
        timestamp_ms,
        pose
):
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame
    )
    result = pose.detect_for_video(
        mp_image,
        timestamp_ms
    )

    if not result.pose_landmarks:
        return None
    landmarks = result.pose_landmarks[0]

    def point(index):
        return np.array([
            landmarks[index].x,
            landmarks[index].y
        ])

    return {
        "left_wrist": point(15),
        "right_wrist": point(16),
        "left_ankle": point(27),
        "right_ankle": point(28),
    }


# Detect body keypoints from uploaded exercise video
def detect_body_keypoints(file: str):
    model = YOLO(MODEL_PATH)
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MP_MODEL_PATH
        ),
        running_mode=VisionRunningMode.VIDEO
    )
    pose = PoseLandmarker.create_from_options(
        options
    )
    cap = cv2.VideoCapture(file)
    fps = (
            cap.get(cv2.CAP_PROP_FPS)
            or 30
    )
    keypoints_per_frame = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
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
            timestamp_ms = int(
                (frame_id / fps) * 1000
            )
            mp_points = get_mp_points(
                frame,
                timestamp_ms,
                pose
            )
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
                ],
                "mediapipe_points": (
                    {
                        k: v.tolist()
                        for k, v in mp_points.items()
                    }
                    if mp_points
                    else None
                )
            }
            keypoints_per_frame.append(
                frame_data
            )
            detected = True
            break

        if not detected:
            continue

    cap.release()
    pose.close()

    # Validation
    if len(keypoints_per_frame) == 0:
        raise KeypointNotDetectedException()

    return keypoints_per_frame