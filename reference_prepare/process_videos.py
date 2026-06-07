# pose_angle_estimator_yolo_mediapipe_tasks.py

import cv2
import numpy as np
import json
import subprocess
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from app.config.supabase_client import supabase
import os


# =========================
# CONFIG
# =========================
MODEL_PATH = "yolo11m-pose.pt"
MP_MODEL_PATH = "pose_landmarker_lite.task"

CONF_THRES = 0.3
KP_CONF_THRES = 0.15

# =========================
# YOLO KEYPOINT INDEX
# =========================
KP = {
    "ls": 5, "rs": 6,
    "le": 7, "re": 8,
    "lw": 9, "rw": 10,
    "lh": 11, "rh": 12,
    "lk": 13, "rk": 14,
    "la": 15, "ra": 16
}

KP_NAME = {
    0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder",
    7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist",
    11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee",
    15: "left_ankle", 16: "right_ankle"
}

ANGLE_TO_KP = {
    "right_shoulder": KP["rs"],
    "left_shoulder": KP["ls"],
    "right_elbow": KP["re"],
    "left_elbow": KP["le"],
    "right_hip": KP["rh"],
    "left_hip": KP["lh"],
    "right_knee": KP["rk"],
    "left_knee": KP["lk"],
}

# =========================
# MEDIAPIPE TASKS INIT
# =========================
BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MP_MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO
)

pose = PoseLandmarker.create_from_options(options)

def upload_video_to_supabase(video_path):
    filename = os.path.basename(video_path).replace(" ", "_")

    bucket = supabase.storage.from_("AngleSync_Project")

    # 🔥 ลบไฟล์เก่าก่อน (ถ้ามี)
    try:
        bucket.remove([filename])
    except:
        pass  # ไม่มีไฟล์ก็ข้าม

    # 🔥 อัปโหลดใหม่
    with open(video_path, "rb") as f:
        bucket.upload(
            path=filename,
            file=f,
            file_options={"content-type": "video/mp4"}
        )

    return bucket.get_public_url(filename)
# DOWNLOAD VIDEO
# =========================
def download_video(url, output="input.mp4"):
    cmd = [
        "yt-dlp",
        "-o", output,
        "--download-sections", "*00:00:18-00:00:23",
        "--force-keyframes-at-cuts",
        "--no-overwrites",
        url
    ]
    subprocess.run(cmd)
    return output

# =========================
# ANGLE
# =========================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)

    ang = np.degrees(
        np.arctan2(c[1]-b[1], c[0]-b[0]) -
        np.arctan2(a[1]-b[1], a[0]-b[0])
    )

    ang = abs(ang)
    if ang > 180:
        ang = 360 - ang

    return float(ang)

# =========================
# SELECT PERSON (FIXED)
# =========================
def select_person(r):
    if r.keypoints is None:
        return None

    if r.keypoints.xy is None:
        return None

    num_people = len(r.keypoints.xy)
    if num_people == 0:
        return None

    # 🔥 ไม่มี confidence → เลือกคนแรก
    if r.keypoints.conf is None:
        return 0

    best_idx = 0
    best_score = -1

    for i in range(num_people):
        conf = r.keypoints.conf[i]
        if conf is None:
            continue

        score = (conf > KP_CONF_THRES).sum().item()

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx

# =========================
# MEDIAPIPE KEYPOINTS
# =========================
def get_mp_points(frame, timestamp_ms, pose):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = pose.detect_for_video(mp_image, timestamp_ms)

    if not result.pose_landmarks:
        return None

    lm = result.pose_landmarks[0]

    def pt(i):
        return np.array([lm[i].x, lm[i].y])

    return {
        "lw": pt(15),
        "rw": pt(16),
        "la": pt(27),
        "ra": pt(28),
    }

# =========================
# ANGLES
# =========================
def get_angles(xyn, mp_pts):
    angles = {}

    try:
        angles["right_shoulder"] = calculate_angle(xyn[12], xyn[6], xyn[8])
        angles["left_shoulder"]  = calculate_angle(xyn[11], xyn[5], xyn[7])

        angles["right_elbow"] = calculate_angle(xyn[6], xyn[8], xyn[10])
        angles["left_elbow"]  = calculate_angle(xyn[5], xyn[7], xyn[9])

        if mp_pts:
            angles["right_wrist"] = calculate_angle(xyn[8], xyn[10], mp_pts["rw"])
            angles["left_wrist"]  = calculate_angle(xyn[7], xyn[9], mp_pts["lw"])

        angles["right_hip"] = calculate_angle(xyn[6], xyn[12], xyn[14])
        angles["left_hip"]  = calculate_angle(xyn[5], xyn[11], xyn[13])

        angles["right_knee"] = calculate_angle(xyn[12], xyn[14], xyn[16])
        angles["left_knee"]  = calculate_angle(xyn[11], xyn[13], xyn[15])

        if mp_pts:
            angles["right_ankle"] = calculate_angle(xyn[14], xyn[16], mp_pts["ra"])
            angles["left_ankle"]  = calculate_angle(xyn[13], xyn[15], mp_pts["la"])

    except:
        return {}

    return {k: round(float(v), 1) for k, v in angles.items()}

# =========================
# MAIN
# =========================
def process(video_path):

    # =========================
    # 🔥 INIT MODEL (YOLO)
    # =========================
    model = YOLO(MODEL_PATH)

    # =========================
    # 🔥 INIT MEDIAPIPE (ต้องอยู่ใน function!)
    # =========================
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MP_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO
    )

    pose = PoseLandmarker.create_from_options(options)

    # =========================
    # VIDEO
    # =========================
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    angles_data = []
    keypoints_data = []

    frame_id = 0

    # =========================
    # LOOP
    # =========================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        h, w = frame.shape[:2]

        results = model(frame, conf=CONF_THRES, verbose=False)

        for r in results:
            idx = select_person(r)
            if idx is None:
                continue

            xyn = r.keypoints.xyn[idx].cpu().numpy()

            # 🔥 กัน None
            if r.keypoints.conf is not None:
                conf = r.keypoints.conf[idx].cpu().numpy()
            else:
                conf = np.ones(len(xyn))

            # =========================
            # 🔥 FIX TIMESTAMP (สำคัญมาก)
            # =========================
            timestamp_ms = int((frame_id / fps) * 1000)

            # 🔥 ส่ง pose เข้าไป
            mp_pts = get_mp_points(frame, timestamp_ms, pose)

            angles = get_angles(xyn, mp_pts)

            # =========================
            # DRAW
            # =========================
            for name, val in angles.items():
                if name not in ANGLE_TO_KP:
                    continue

                kp_idx = ANGLE_TO_KP[name]
                pt = xyn[kp_idx]

                x, y = int(pt[0]*w), int(pt[1]*h)

                cv2.putText(frame, str(int(val)),
                            (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,255,0), 2)

            # =========================
            # SAVE KEYPOINTS
            # =========================
            kp_frame = {
                "frame": frame_id,
                "time": round(frame_id/fps, 2),
                "keypoints": []
            }

            for i in range(len(xyn)):
                kp_frame["keypoints"].append({
                    "id": i,
                    "name": KP_NAME.get(i, "unknown"),
                    "x": float(xyn[i][0]),
                    "y": float(xyn[i][1]),
                    "conf": float(conf[i])
                })

            keypoints_data.append(kp_frame)

            # =========================
            # SAVE ANGLES
            # =========================
            angles_data.append({
                "frame": frame_id,
                "time": round(frame_id/fps, 2),
                **angles
            })

            break

        cv2.imshow("YOLO + MediaPipe", frame)
        if cv2.waitKey(1) == 27:
            break

    # =========================
    # CLEANUP
    # =========================
    cap.release()
    cv2.destroyAllWindows()

    # =========================
    # SAVE JSON
    # =========================
    with open("output_angles.json", "w") as f:
        json.dump(angles_data, f, indent=2)

    with open("output_keypoints.json", "w") as f:
        json.dump(keypoints_data, f, indent=2)

    print("Saved output_angles.json")
    print("Saved output_keypoints.json")

    # =========================
    # SAVE TO DB
    # =========================
    save_to_db(video_path, angles_data, keypoints_data)

def save_to_db(video_path, angles_data, keypoints_data):

    # =========================
    # 1. upload video
    # =========================
    video_url = upload_video_to_supabase(video_path)

    gender = "male" if "men" in video_path else "female"

    # =========================
    # 2. insert video
    # =========================
    res = supabase.table("exercise_reference").insert({
        "exercise_name": os.path.basename(video_path),
        "reference_video_url": video_url,
        "reference_gender": gender
    }).execute()

    video_id = res.data[0]["reference_video_id"]

    # =========================
    # 3. insert frame + pose
    # =========================
    for i in range(len(angles_data)):

        frame = angles_data[i]
        kp = keypoints_data[i]

        frame_res = supabase.table("exercise_reference_frame_data").insert({
            "parent_video_id": video_id,
            "frame_sequence": frame["frame"],
            "frame_time_sec": frame["time"]
        }).execute()

        frame_id = frame_res.data[0]["reference_frame_id"]

        supabase.table("exercise_reference_pose_metrics").insert({
            "related_frame_id": frame_id,
            "joint_angle_data": frame,
            "joint_coordinate_data": kp["keypoints"]
        }).execute()

    print(f"✅ Saved to DB: {video_path}")
# =========================
# RUN
# =========================
if __name__ == "__main__":
    import sys

    src = sys.argv[1]

    if src.startswith("http"):
        path = download_video(src)
    else:
        path = src

    process(path)