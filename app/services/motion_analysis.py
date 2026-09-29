import numpy as np
from app.exceptions import (
    ExerciseMismatchException,
    InvalidKeypointsException
)

ANGLE_METADATA_KEYS = {"frame", "time"}
REFERENCE_SAMPLE_COUNT = 40
MIN_EXERCISE_SIMILARITY = 50.0
MAX_MISMATCH_ANGLE_ERROR = 32.0
MAX_MISMATCH_RANGE_ERROR = 38.0

# percentile ที่ใช้ตัด outlier ก่อนคำนวณ range ของมุมข้อต่อ
# (กันกรณี pose detection หลุดไป 1-2 เฟรม เช่น ตอนเข้า/ออกท่า หรือกล้องสั่น
# ทำให้ range พุ่งผิดปกติทั้งที่ตลอดคลิปจริง ๆ ท่านิ่ง)
RANGE_LOWER_PERCENTILE = 5
RANGE_UPPER_PERCENTILE = 95

# สัดส่วนที่ตัดทิ้งจากต้น/ท้าย sequence ก่อนคำนวณ range สำหรับ rule_check ของ
# ท่าที่ต้องการเช็ค "นิ่งพอ" (เช่น plank) ช่วงต้น/ท้ายคลิปมักเป็นตอนคนขยับ
# เข้า-ออกจากท่า ซึ่งเป็นการเคลื่อนไหวจริงไม่ใช่ noise แต่ไม่ควรถูกนับเป็น
# ส่วนหนึ่งของ "ท่าที่ hold" ตอนเช็ค stability
RULE_CHECK_TRIM_RATIO = 0.15
RULE_CHECK_MIN_FRAMES_AFTER_TRIM = 5


# Calculate angle between 3 points using vector analysis
def calculate_angle(start_point, middle_point, end_point):
    start_point = np.array(start_point)
    middle_point = np.array(middle_point)
    end_point = np.array(end_point)

    vector_start = start_point - middle_point
    vector_end = end_point - middle_point

    ba_norm = np.linalg.norm(vector_start)
    bc_norm = np.linalg.norm(vector_end)
    if ba_norm == 0 or bc_norm == 0:
        return 0
    cosine_angle = np.dot(vector_start, vector_end) / (
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


def _angle_only(frame_angles):
    return {
        joint_name: float(value)
        for joint_name, value in frame_angles.items()
        if joint_name not in ANGLE_METADATA_KEYS
    }


def _common_joints(first_sequence, second_sequence):
    first_joints = set(first_sequence[0].keys())
    second_joints = set(second_sequence[0].keys())

    return sorted(
        first_joints.intersection(second_joints)
    )


def _sequence_matrix(angle_sequence, joints):
    return np.array(
        [
            [
                float(frame[joint])
                for joint in joints
            ]
            for frame in angle_sequence
        ],
        dtype=float
    )


def _resample_matrix(matrix, sample_count=REFERENCE_SAMPLE_COUNT):
    if len(matrix) == sample_count:
        return matrix

    if len(matrix) == 1:
        return np.repeat(
            matrix,
            sample_count,
            axis=0
        )

    source_x = np.linspace(
        0.0,
        1.0,
        num=len(matrix)
    )
    target_x = np.linspace(
        0.0,
        1.0,
        num=sample_count
    )

    resampled = [
        np.interp(
            target_x,
            source_x,
            matrix[:, joint_index]
        )
        for joint_index in range(matrix.shape[1])
    ]

    return np.stack(
        resampled,
        axis=1
    )


def _movement_signature(matrix):
    return np.ptp(
        matrix,
        axis=0
    )


def _dtw_alignment(
        user_matrix,
        reference_matrix
):
    user_length = len(user_matrix)
    reference_length = len(reference_matrix)

    costs = np.full(
        (
            user_length + 1,
            reference_length + 1
        ),
        np.inf
    )
    costs[0, 0] = 0.0

    distances = np.zeros(
        (
            user_length,
            reference_length
        )
    )

    for user_index in range(user_length):
        for reference_index in range(reference_length):
            distances[user_index, reference_index] = np.mean(
                np.abs(
                    user_matrix[user_index]
                    - reference_matrix[reference_index]
                )
            )
            costs[user_index + 1, reference_index + 1] = (
                    distances[user_index, reference_index]
                    + min(
                costs[user_index, reference_index + 1],
                costs[user_index + 1, reference_index],
                costs[user_index, reference_index]
            )
            )

    path = []
    user_index = user_length
    reference_index = reference_length

    while (
            user_index > 0
            and reference_index > 0
    ):
        path.append(
            (
                user_index - 1,
                reference_index - 1
            )
        )
        previous_steps = (
            costs[user_index - 1, reference_index],
            costs[user_index, reference_index - 1],
            costs[user_index - 1, reference_index - 1]
        )
        step = int(
            np.argmin(previous_steps)
        )

        if step == 0:
            user_index -= 1
        elif step == 1:
            reference_index -= 1
        else:
            user_index -= 1
            reference_index -= 1

    path.reverse()

    normalized_cost = costs[user_length, reference_length] / max(
        len(path),
        1
    )

    return float(normalized_cost), path


MAX_SHIFTS = 8


def _best_circular_dtw_alignment(user_matrix, reference_matrix):
    best_cost = None
    best_shift = 0
    best_path = []

    # sample แค่ 8 shifts แทน 40
    total = len(reference_matrix)
    shifts = [int(i * total / MAX_SHIFTS) for i in range(MAX_SHIFTS)]

    for shift in shifts:
        shifted_reference = np.roll(reference_matrix, shift, axis=0)
        cost, path = _dtw_alignment(user_matrix, shifted_reference)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_shift = shift
            best_path = path

    return best_cost, best_shift, best_path


def _align_reference_to_user(
        user_matrix,
        reference_matrix
):
    reference_matrix = _resample_matrix(
        reference_matrix,
        sample_count=len(user_matrix)
    )

    _, best_shift, path = _best_circular_dtw_alignment(
        user_matrix,
        reference_matrix
    )
    shifted_reference = np.roll(
        reference_matrix,
        best_shift,
        axis=0
    )

    aligned_reference = []
    for user_index in range(len(user_matrix)):
        matched_reference_indexes = [
            reference_index
            for path_user_index, reference_index in path
            if path_user_index == user_index
        ]

        if matched_reference_indexes:
            aligned_reference.append(
                np.mean(
                    shifted_reference[matched_reference_indexes],
                    axis=0
                )
            )
        else:
            aligned_reference.append(
                shifted_reference[
                    min(
                        user_index,
                        len(shifted_reference) - 1
                    )
                ]
            )

    return np.array(aligned_reference), best_shift


def _angle_range(sequence, joints):
    """
    หา range ของแต่ละ joint ตลอด sequence แบบ max - min ตรง ๆ
    (พฤติกรรมดั้งเดิม ใช้กับ push/squat/split/lunge ที่ต้องเช็คว่า
    "ขยับมากพอ" — lower-bound — จึงต้องการ range เต็มที่ไม่ถูกตัดทิ้ง)
    """
    ranges = {}

    for joint in joints:
        values = [
            frame[joint]
            for frame in sequence
            if joint in frame
        ]
        if values:
            ranges[joint] = max(values) - min(values)

    return ranges


def _angle_range_stable(sequence, joints):
    """
    หา range ของแต่ละ joint แบบตัด outlier ทิ้งก่อน (percentile 5-95
    แทน max-min ตรง ๆ) ใช้เฉพาะกับท่าที่ต้องเช็คว่า "นิ่งพอ" (upper-bound
    เช่น plank) เพื่อไม่ให้ 1-2 เฟรมที่ pose detection หลุด (ตอนเข้า/ออกท่า,
    กล้องสั่น) ทำให้ range พุ่งผิดปกติทั้งที่ตลอดคลิปจริง ๆ ท่านิ่งมาก

    ฟังก์ชันนี้แยกจาก _angle_range เดิมโดยเจตนา เพื่อไม่ให้ push/squat/
    split ที่ใช้ _angle_range ปกติถูกกระทบ
    """
    ranges = {}

    for joint in joints:
        values = [
            frame[joint]
            for frame in sequence
            if joint in frame
        ]
        if values:
            if len(values) >= 5:
                low = np.percentile(values, RANGE_LOWER_PERCENTILE)
                high = np.percentile(values, RANGE_UPPER_PERCENTILE)
            else:
                # ข้อมูลน้อยเกินจะตัด percentile ให้มีความหมาย ใช้ max-min ตรง ๆ แทน
                low = min(values)
                high = max(values)
            ranges[joint] = float(high - low)

    return ranges


def _trim_sequence_edges(
        sequence,
        trim_ratio=RULE_CHECK_TRIM_RATIO,
        min_frames_after_trim=RULE_CHECK_MIN_FRAMES_AFTER_TRIM
):
    """
    ตัด frame ช่วงต้น/ท้ายของ sequence ทิ้งก่อนคำนวณ range สำหรับ rule_check
    เพื่อไม่ให้ช่วง transition เข้า-ออกท่า (เช่น ก้มตัวลงเข้า plank,
    ลุกขึ้นตอนจบ) ไปปนกับช่วงที่ hold ท่าจริง ๆ

    ถ้าตัดแล้วเหลือ frame น้อยเกินไป (สั้นเกินกว่าจะเชื่อถือได้)
    จะคืน sequence เดิมทั้งหมดแทนที่จะตัด
    """
    total_frames = len(sequence)
    trim_count = int(total_frames * trim_ratio)

    if total_frames - (2 * trim_count) < min_frames_after_trim:
        return sequence

    return sequence[trim_count: total_frames - trim_count]


def _exercise_rule_check(
        exercise_name,
        detected_angle_sequence,
        comparison
):
    name = (exercise_name or "").lower()

    # push/squat/split ใช้ range ดั้งเดิม (max-min ตรง ๆ, ไม่ตัดอะไรเลย)
    # เพราะต้องเช็คว่า "ขยับมากพอ" (lower-bound) — พฤติกรรมเดียวกับ
    # ก่อนมีการแก้ปัญหา plank ทุกประการ
    full_ranges = _angle_range(
        detected_angle_sequence,
        comparison["compared_joints"]
    )

    # plank ต้องการเช็คว่า "นิ่งพอ" (upper-bound) จึงตัด transition
    # เข้า/ออกท่า (_trim_sequence_edges) และตัด outlier ด้วย percentile
    # (_angle_range_stable) ก่อนคำนวณ range — แยกฟังก์ชันจากท่าอื่น
    # โดยเจตนา ไม่กระทบ full_ranges ด้านบนเลย
    trimmed_sequence = _trim_sequence_edges(detected_angle_sequence)
    trimmed_ranges = _angle_range_stable(
        trimmed_sequence,
        comparison["compared_joints"]
    )

    def max_range(ranges, *joint_names):
        values = [
            ranges[joint_name]
            for joint_name in joint_names
            if joint_name in ranges
        ]

        if not values:
            return 0.0

        return max(values)

    upper_body_range = max_range(
        full_ranges,
        "left_elbow",
        "right_elbow",
        "left_shoulder",
        "right_shoulder"
    )
    lower_body_range = max_range(
        full_ranges,
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee"
    )

    if "push" in name:
        passed = (
                upper_body_range >= 10.0
        )
        return {
            "passed": passed,
            "reason": "push_up_upper_body_motion",
            "upper_body_range": round(float(upper_body_range), 2),
            "lower_body_range": round(float(lower_body_range), 2)
        }

    if "squat" in name:
        passed = (
                lower_body_range >= 12.0
        )
        return {
            "passed": passed,
            "reason": "squat_lower_body_motion",
            "upper_body_range": round(float(upper_body_range), 2),
            "lower_body_range": round(float(lower_body_range), 2)
        }

    if (
            "split" in name
            or "jump" in name
            or "lunge" in name
    ):
        passed = (
                lower_body_range >= 18.0
                and lower_body_range >= upper_body_range
        )
        return {
            "passed": passed,
            "reason": "split_jump_lower_body_motion",
            "upper_body_range": round(float(upper_body_range), 2),
            "lower_body_range": round(float(lower_body_range), 2)
        }

    if "plank" in name:
        plank_upper_body_range = max_range(
            trimmed_ranges,
            "left_elbow",
            "right_elbow",
            "left_shoulder",
            "right_shoulder"
        )
        plank_lower_body_range = max_range(
            trimmed_ranges,
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee"
        )
        passed = (
                plank_lower_body_range <= 45.0
                and plank_upper_body_range <= 55.0
        )
        return {
            "passed": passed,
            "reason": "plank_stability",
            "upper_body_range": round(float(plank_upper_body_range), 2),
            "lower_body_range": round(float(plank_lower_body_range), 2)
        }

    return {
        "passed": True,
        "reason": "no_specific_rule",
        "upper_body_range": round(float(upper_body_range), 2),
        "lower_body_range": round(float(lower_body_range), 2)
    }


def compare_exercise_sequences(
        detected_angle_sequence,
        reference_angle_sequence
):
    user_sequence = [
        _angle_only(frame)
        for frame in detected_angle_sequence
    ]
    reference_sequence = [
        _angle_only(frame)
        for frame in reference_angle_sequence
    ]

    if (
            not user_sequence
            or not reference_sequence
    ):
        raise InvalidKeypointsException()

    joints = _common_joints(
        user_sequence,
        reference_sequence
    )

    if not joints:
        raise InvalidKeypointsException()

    user_matrix = _resample_matrix(
        _sequence_matrix(
            user_sequence,
            joints
        )
    )
    reference_matrix = _resample_matrix(
        _sequence_matrix(
            reference_sequence,
            joints
        )
    )

    angle_error, best_phase_shift, _ = _best_circular_dtw_alignment(
        user_matrix,
        reference_matrix
    )

    range_error = np.mean(
        np.abs(
            _movement_signature(user_matrix)
            - _movement_signature(reference_matrix)
        )
    )

    signature_similarity = max(
        0.0,
        100.0 - (
                (angle_error * 0.65)
                + (range_error * 0.35)
        )
    )

    return {
        "similarity_score": round(
            float(signature_similarity),
            2
        ),
        "average_angle_error": round(
            float(angle_error),
            2
        ),
        "movement_range_error": round(
            float(range_error),
            2
        ),
        "phase_shift": int(best_phase_shift),
        "compared_joints": joints
    }


def validate_exercise_match(
        detected_angle_sequence,
        reference_angle_sequence,
        exercise_name=None
):
    comparison = compare_exercise_sequences(
        detected_angle_sequence,
        reference_angle_sequence
    )

    has_low_similarity = (
            comparison["similarity_score"]
            < MIN_EXERCISE_SIMILARITY
    )
    has_different_motion_pattern = (
            comparison["average_angle_error"]
            > MAX_MISMATCH_ANGLE_ERROR
            and comparison["movement_range_error"]
            > MAX_MISMATCH_RANGE_ERROR
    )
    rule_result = _exercise_rule_check(
        exercise_name,
        detected_angle_sequence,
        comparison
    )
    comparison["rule_check"] = rule_result

    if (
            has_low_similarity
            or has_different_motion_pattern
            or not rule_result["passed"]
    ):
        raise ExerciseMismatchException(
            similarity_score=comparison["similarity_score"],
            details={
                "average_angle_error": comparison["average_angle_error"],
                "movement_range_error": comparison["movement_range_error"],
                "rule_check": rule_result
            }
        )

    return comparison


def calculate_sequence_risk_scores(
        detected_angle_sequence,
        reference_angle_sequence
):
    user_sequence = [
        _angle_only(frame)
        for frame in detected_angle_sequence
    ]
    reference_sequence = [
        _angle_only(frame)
        for frame in reference_angle_sequence
    ]

    joints = _common_joints(
        user_sequence,
        reference_sequence
    )

    user_matrix = _sequence_matrix(
        user_sequence,
        joints
    )
    reference_matrix = _resample_matrix(
        _sequence_matrix(
            reference_sequence,
            joints
        ),
        sample_count=len(user_matrix)
    )
    reference_matrix, _ = _align_reference_to_user(
        user_matrix,
        reference_matrix
    )

    return [
        float(
            min(
                np.mean(
                    np.abs(
                        user_matrix[index]
                        - reference_matrix[index]
                    )
                ),
                100.0
            )
        )
        for index in range(len(user_matrix))
    ]


# Analyze motion quality using detected body keypoints
def analyze_motion(
        keypoints_per_frame: tuple,
        standard_angle: dict,
        reference_angle_sequence=None,
        exercise_name=None
):
    if (
            not keypoints_per_frame
            or len(keypoints_per_frame) == 0
    ):
        raise InvalidKeypointsException()

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

        angles_per_frame.append({
            "frame": frame_data["frame"],
            "time": frame_data["time"],
            "angles": angles,
            "keypoints": keypoints,
            "risk_score": 0.0
        })

        valid_frame_count += 1

    if valid_frame_count == 0:
        raise InvalidKeypointsException()

    detected_angle_sequence = [
        frame["angles"]
        for frame in angles_per_frame
    ]

    exercise_match = None

    if reference_angle_sequence:
        exercise_match = validate_exercise_match(
            detected_angle_sequence,
            reference_angle_sequence,
            exercise_name
        )
        risk_scores = calculate_sequence_risk_scores(
            detected_angle_sequence,
            reference_angle_sequence
        )
    else:
        risk_scores = [
            calculate_risk_score(
                angles,
                standard_angle
            )
            for angles in detected_angle_sequence
        ]

    for index, risk_score in enumerate(risk_scores):
        angles_per_frame[index]["risk_score"] = risk_score

    peak_search_count = int(
        np.ceil(len(risk_scores) * 0.95)
    )
    peak_search_count = max(
        1,
        peak_search_count
    )

    highest_risk_frame_index = int(
        np.argmax(risk_scores[:peak_search_count])
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
            angles_per_frame,

        "exercise_match":
            exercise_match
    }