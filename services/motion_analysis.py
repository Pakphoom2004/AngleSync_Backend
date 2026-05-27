import numpy as np
from exceptions import InvalidKeypointsException

# Calculate angle between 3 points using vector analysis
def calculate_angle(first_angle, second_angle, third_angle):
    first_angle = np.array(first_angle)
    second_angle = np.array(second_angle)
    third_angle = np.array(third_angle)

    angle_one = first_angle - second_angle
    angle_two = third_angle - second_angle

    ba_norm = np.linalg.norm(angle_one)
    bc_norm = np.linalg.norm(angle_two)
    if ba_norm == 0 or bc_norm == 0:
        return 0
    cosine_angle = np.dot(angle_one, angle_two) / (
            ba_norm * bc_norm
    )

    cosine_angle = np.clip(
        cosine_angle,
        -1.0,
        1.0
    )

    angle = np.degrees(
        np.arccos(cosine_angle)
    )

    return float(angle)

# Extract joint angles from detected keypoints
def extract_joint_angles(frame_data):
    keypoints = frame_data["yolo_keypoints"]

    kp = {}

    for point in keypoints:
        kp[point["name"]] = (
            point["x"],
            point["y"]
        )

    angles = {}

    try:
        angles["left_elbow"] = calculate_angle(
            kp["left_shoulder"],
            kp["left_elbow"],
            kp["left_wrist"]
        )

        angles["right_elbow"] = calculate_angle(
            kp["right_shoulder"],
            kp["right_elbow"],
            kp["right_wrist"]
        )

        angles["left_shoulder"] = calculate_angle(
            kp["left_hip"],
            kp["left_shoulder"],
            kp["left_elbow"]
        )

        angles["right_shoulder"] = calculate_angle(
            kp["right_hip"],
            kp["right_shoulder"],
            kp["right_elbow"]
        )

        angles["left_knee"] = calculate_angle(
            kp["left_hip"],
            kp["left_knee"],
            kp["left_ankle"]
        )

        angles["right_knee"] = calculate_angle(
            kp["right_hip"],
            kp["right_knee"],
            kp["right_ankle"]
        )

        angles["left_hip"] = calculate_angle(
            kp["left_shoulder"],
            kp["left_hip"],
            kp["left_knee"]
        )

        angles["right_hip"] = calculate_angle(
            kp["right_shoulder"],
            kp["right_hip"],
            kp["right_knee"]
        )

    except KeyError:
        return None

    return angles

# Calculate movement risk score from angle deviation
def calculate_risk_score(
        detected_angles,
        standard_angle
):
    total_deviation = 0
    compared_joint_count = 0

    for joint_name, standard_value in standard_angle.items():

        if joint_name not in detected_angles:
            continue

        detected_value = detected_angles[
            joint_name
        ]

        deviation = abs(
            detected_value - standard_value
        )

        total_deviation += deviation
        compared_joint_count += 1

    if compared_joint_count == 0:
        return 100.0

    average_deviation = (
        total_deviation
        / compared_joint_count
    )

    risk_score = min(
        average_deviation,
        100.0
    )

    return float(risk_score)

# Analyze motion quality using detected body keypoints
def analyze_motion(
        keypoints_per_frame: tuple,
        standard_angle: dict
):
    if (
        not keypoints_per_frame
        or len(keypoints_per_frame) == 0
    ):
        raise InvalidKeypointsException()

    risk_scores = []
    angles_per_frame = []
    valid_frame_count = 0

    for frame_data in keypoints_per_frame:

        keypoints = frame_data[
            "yolo_keypoints"
        ]

        confidence_sum = sum(
            point["confidence"]
            for point in keypoints
        )

        if confidence_sum == 0:
            continue

        angles = extract_joint_angles(
            frame_data
        )

        if angles is None:
            continue

        risk_score = calculate_risk_score(
            angles,
            standard_angle
        )

        risk_scores.append(risk_score)

        angles_per_frame.append({
            "frame": frame_data["frame"],
            "time": frame_data["time"],
            "angles": angles,
            "risk_score": risk_score
        })

        valid_frame_count += 1

    if valid_frame_count == 0:
        raise InvalidKeypointsException()

    highest_risk_frame_index = int(
        np.argmax(risk_scores)
    )

    average_risk = np.mean(
        risk_scores
    )

    accuracy_score = max(
        0,
        100 - average_risk
    )

    return {
        "accuracy_score": round(
            float(accuracy_score),
            2
        ),

        "risk_scores": tuple(
            round(score, 2)
            for score in risk_scores
        ),

        "highest_risk_frame_index":
            highest_risk_frame_index,

        "angles_per_frame":
            angles_per_frame
    }
