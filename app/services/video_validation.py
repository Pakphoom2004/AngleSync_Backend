import os

import cv2

from app.exceptions import ServiceException


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
        raise ServiceException("Video exceeds 60 seconds.")
