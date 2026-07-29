import os
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import io
import uuid
from PIL import Image

from app.config.supabase_client import supabase

STORAGE_BUCKET = "AngleSync_Project"
STORAGE_FOLDER = "risk_frames"


def _open_video_capture(file: str):
    cap = cv2.VideoCapture(file)
    orientation_auto = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    if orientation_auto is not None:
        cap.set(orientation_auto, 1)
    return cap


def generate_risk_graph(
        risk_scores: tuple,
        highest_risk_frame_index: int,
        frame_times=None,
        fps: int = 30
):
    if frame_times:
        time_axis = [
            round(float(time), 2)
            for time in frame_times
        ]
    else:
        time_axis = [
            round(frame / fps, 2)
            for frame in range(len(risk_scores))
        ]

    highest_risk_score = risk_scores[
        highest_risk_frame_index
    ]

    highest_risk_time = time_axis[
        highest_risk_frame_index
    ]

    plt.figure(figsize=(12, 6))

    plt.plot(
        time_axis,
        risk_scores,
        linewidth=2.5,
        color="#36B66B",
        label="Risk score"
    )

    plt.scatter(
        highest_risk_time,
        highest_risk_score,
        s=220,
        color="#FF3B30",
        edgecolors="#111111",
        linewidths=1.5,
        zorder=5,
        label="Highest risk"
    )

    plt.axvline(
        highest_risk_time,
        color="#FF3B30",
        linestyle="--",
        linewidth=1.4,
        alpha=0.8
    )

    plt.annotate(
        f"Highest risk\n{highest_risk_score:.1f}%",
        xy=(
            highest_risk_time,
            highest_risk_score
        ),
        xytext=(12, 18),
        textcoords="offset points",
        arrowprops={
            "arrowstyle": "->",
            "color": "#FF3B30",
            "lw": 1.5
        },
        fontsize=11,
        fontweight="bold",
        color="#111111"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Risk Score (%)")
    plt.title("Movement Risk Analysis")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    return Image.open(buf)


def save_highest_risk_frame(
        video_path: str,
        frame_number: int,
        keypoints=None
):
    cap = _open_video_capture(video_path)

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    except (TypeError, ValueError):
        total_frames = 0

    target_frame = max(int(frame_number), 1)
    if total_frames > 0:
        target_frame = min(target_frame, total_frames)

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        target_frame - 1
    )

    success, frame = cap.read()

    if not success:
        cap.release()
        cap = _open_video_capture(video_path)
        last_frame = None

        for current_frame in range(1, target_frame + 1):
            read_success, candidate_frame = cap.read()
            if not read_success:
                break
            last_frame = candidate_frame
            if current_frame == target_frame:
                success = True
                frame = candidate_frame
                break

        if not success and last_frame is not None:
            success = True
            frame = last_frame

    if not success:
        cap.release()
        raise Exception(
            f"Failed to extract frame. Requested frame={frame_number}, "
            f"target frame={target_frame}, total frames={total_frames}."
        )
    if keypoints:
        frame = draw_skeleton_on_frame(
            frame,
            keypoints
        )

    encode_success, buffer = cv2.imencode(".jpg", frame)
    cap.release()

    if not encode_success:
        raise Exception("Failed to encode highest risk frame image.")

    filename = f"highest_risk_frame_{uuid.uuid4().hex}.jpg"
    storage_path = f"{STORAGE_FOLDER}/{filename}"
    image_bytes = buffer.tobytes()

    supabase.storage.from_(STORAGE_BUCKET).upload(
        storage_path,
        image_bytes,
        {"content-type": "image/jpeg"}
    )

    public_url = supabase.storage.from_(STORAGE_BUCKET).get_public_url(storage_path)

    return public_url


SKELETON_CONNECTIONS = [
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle")
]


def draw_skeleton_on_frame(frame, keypoints):
    height, width = frame.shape[:2]
    points = {}

    for point in keypoints:
        if point.get("confidence", 1.0) <= 0:
            continue

        points[point["name"]] = (
            int(point["x"] * width),
            int(point["y"] * height)
        )

    for start_name, end_name in SKELETON_CONNECTIONS:
        if (
            start_name not in points
            or end_name not in points
        ):
            continue

        cv2.line(
            frame,
            points[start_name],
            points[end_name],
            (0, 210, 90),
            4
        )

    for point in points.values():
        cv2.circle(
            frame,
            point,
            6,
            (0, 0, 255),
            -1
        )
        cv2.circle(
            frame,
            point,
            8,
            (255, 255, 255),
            2
        )

    return frame