import cv2
import numpy as np

from exceptions import (
    KeypointNotDetectedException,
    ServiceException
)

MODEL_PATH = "assets/yolo11m-pose.pt"

CONF_THRES = 0.3
KP_CONF_THRES = 0.15

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


# Detect body keypoints from uploaded exercise video
def detect_body_keypoints(file: str):
    yolo_model = _load_pose_dependencies()

    model = yolo_model(MODEL_PATH)

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

    # Validation
    if len(keypoints_per_frame) == 0:
        raise KeypointNotDetectedException()

    return keypoints_per_frame
