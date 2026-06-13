import cv2
import numpy as np

SKELETON_CONNECTIONS = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (11, 12),
    (5, 11),
    (6, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16)
]


def render_skeleton_overlay(
        frame,
        keypoints: tuple
):
    skeleton_overlay = frame.copy()

    height, width = skeleton_overlay.shape[:2]

    points = []

    for point in keypoints:
        x = int(point[0] * width)
        y = int(point[1] * height)
        points.append((x, y))

    for start_idx, end_idx in SKELETON_CONNECTIONS:
        x1, y1 = points[start_idx]
        x2, y2 = points[end_idx]

        cv2.line(
            skeleton_overlay,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

    for x, y in points:
        cv2.circle(
            skeleton_overlay,
            (x, y),
            5,
            (0, 0, 255),
            -1
        )

    return skeleton_overlay