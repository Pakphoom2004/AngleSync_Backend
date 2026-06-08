from app.services.motion_analysis import (
    analyze_motion,
    extract_joint_angles,
    validate_exercise_match,
)

from app.services.pose_detection import (
    detect_body_keypoints,
    detect_sample_keypoints,
    verify_file_type,      
    verify_video_duration,
)

from app.services.graph_service import (
    generate_risk_graph,
    save_highest_risk_frame
)

from app.services.feedback_service import (
    generate_advanced_feedback
)

from app.exceptions import (
    ExerciseMismatchException,
    ServiceException
)

def process_video_analysis(
        video_path: str,
        reference_video_id: int,
        progress_callback=None,
        base_url="http://localhost:8000"
):
    def report_progress(percent, step, message):
        if progress_callback is not None:
            progress_callback({
                "step": step,
                "message": message,
                "percent": percent
            })

    verify_file_type(video_path)
    verify_video_duration(video_path)

    # STEP 1: Early exercise validation
    report_progress(20, "validating", "Checking exercise type...")

    reference_data = get_reference_angles_from_db(reference_video_id)

    sample_keypoints = detect_sample_keypoints(video_path)

    sample_angle_sequence = []
    for frame in sample_keypoints:
        angles = extract_joint_angles(frame)
        if angles is not None:          # ต้องอยู่ใน for loop
            sample_angle_sequence.append(angles)

    if len(sample_angle_sequence) < 5:  # ต้องอยู่ใน function
        pass
    else:
        try:
            validate_exercise_match(
                sample_angle_sequence,
                reference_data["angle_sequence"],
                reference_data["exercise_name"]
            )
        except ExerciseMismatchException as error:
            return {
                "status": "exercise_mismatch",
                "can_analyze": False,
                "score": 0.0,
                "score_scale": 100,
                "risk_level": "MISMATCH",
                "message": error.message,
                "exercise_match": {
                    "is_match": False,
                    "similarity_score": round(float(error.similarity_score), 2),
                    **error.details
                },
                "feedback": {
                    "form_summary": "This video does not match the selected reference exercise.",
                    "injury_risk": "Analysis was skipped because the movement is a different exercise.",
                    "corrective_cues": "Choose the matching reference exercise and upload again.",
                    "practice_plan": "Record the same exercise as the selected reference before analyzing."
                }
            }
    # STEP 2: Full detection (ผ่าน validate แล้ว)
    report_progress(30, "detecting", "Detecting pose...")

    keypoints_per_frame = detect_body_keypoints(
        video_path,
        progress_callback=progress_callback
    )

    report_progress(70, "analyzing", "Analyzing movement...")

    try:
        analysis_result = analyze_motion(
            keypoints_per_frame,
            reference_data["average_angles"],
            reference_data["angle_sequence"],
            reference_data["exercise_name"]
        )
    except ExerciseMismatchException as error:
        return {
            "status": "exercise_mismatch",
            "can_analyze": False,
            "score": 0.0,
            "score_scale": 100,
            "risk_level": "MISMATCH",
            "message": error.message,
            "exercise_match": {
                "is_match": False,
                "similarity_score": round(float(error.similarity_score), 2),
                **error.details
            },
            "feedback": {
                "form_summary": "This video does not match the selected reference exercise.",
                "injury_risk": "Analysis was skipped.",
                "corrective_cues": "Choose the matching reference exercise and upload again.",
                "practice_plan": "Record the same exercise as the selected reference before analyzing."
            }
        }

    # STEP 3: Post-analysis processing
    accuracy_score = analysis_result["accuracy_score"]
    risk_scores = analysis_result["risk_scores"]
    highest_risk_frame_index = analysis_result["highest_risk_frame_index"]
    angles_per_frame = analysis_result["angles_per_frame"]
    highest_frame_data = angles_per_frame[highest_risk_frame_index]

    report_progress(82, "generating_graph", "Generating risk graph...")

    graph_image_path = generate_risk_graph(
        risk_scores,
        highest_risk_frame_index,
        frame_times=[frame["time"] for frame in angles_per_frame]
    )

    frame_path = save_highest_risk_frame(
        video_path,
        highest_frame_data["frame"],
        highest_frame_data.get("keypoints")
    )

    report_progress(90, "generating_feedback", "Generating feedback...")

    try:
        feedback_result = generate_advanced_feedback(
            highest_frame_data,
            frame_path
        )
    except ServiceException:
        feedback_result = {
            "form_summary": "AI feedback unavailable.",
            "injury_risk": "Gemini API quota exceeded.",
            "corrective_cues": "Please retry later.",
            "practice_plan": "Analyze again after quota reset."
        }

    risk_level = (
        "GOOD" if accuracy_score >= 80
        else "NORMAL" if accuracy_score >= 60
        else "DANGEROUS"
    )

    return {
        "status": "completed",
        "can_analyze": True,
        "score": round(accuracy_score, 2),
        "score_scale": 100,
        "risk_level": risk_level,
        "exercise_match": {
            "is_match": True,
            **(analysis_result.get("exercise_match") or {})
        },
        "selected_frame": {
            "frame": highest_frame_data["frame"],
            "time": round(highest_frame_data["time"], 2),
            "risk": round(highest_frame_data["risk_score"], 2),
            "image": frame_path
        },
        "graph_data": {
            "graph_image": graph_image_path,
            "risk_scores": [round(float(s), 2) for s in risk_scores],
            "frame_times": [round(float(f["time"]), 2) for f in angles_per_frame],
            "highest_risk_frame_index": highest_risk_frame_index,
            "highest_risk_image_url": f"{base_url}/outputs/highest_risk_frame.jpg",
        },
        "feedback": feedback_result
    }

def get_reference_angles_from_db(reference_video_id: int) -> dict:
    from app.config.supabase_client import supabase

    reference = supabase.table("exercise_reference") \
        .select("exercise_name") \
        .eq("reference_video_id", reference_video_id) \
        .limit(1) \
        .execute()

    exercise_name = ""
    if reference.data:
        exercise_name = reference.data[0].get("exercise_name") or ""

    # step 1
    frames = supabase.table("exercise_reference_frame_data") \
        .select("reference_frame_id, frame_sequence") \
        .eq("parent_video_id", reference_video_id) \
        .order("frame_sequence") \
        .execute()

    print(f"DEBUG reference_video_id: {reference_video_id}")
    print(f"DEBUG frames found: {len(frames.data)}")

    if not frames.data:
        raise ValueError(f"No frames found for reference_video_id={reference_video_id}")

    frame_ids = [
        f["reference_frame_id"]
        for f in frames.data
    ]
    frame_order = {
        f["reference_frame_id"]: index
        for index, f in enumerate(frames.data)
    }

    # step 2
    metrics = supabase.table("exercise_reference_pose_metrics") \
        .select("related_frame_id, joint_angle_data") \
        .in_("related_frame_id", frame_ids) \
        .execute()

    print(f"DEBUG metrics found: {len(metrics.data)}")

    if not metrics.data:
        raise ValueError(f"No metrics found for frame_ids={frame_ids[:5]}")

    sorted_metrics = sorted(
        metrics.data,
        key=lambda row: frame_order.get(
            row["related_frame_id"],
            0
        )
    )

    all_angles = [
        row["joint_angle_data"]
        for row in sorted_metrics
    ]

    averaged = {}
    for joint in all_angles[0].keys():
        if joint in {"frame", "time"}:
            continue
        values = [f[joint] for f in all_angles if joint in f]
        averaged[joint] = sum(values) / len(values)

    return {
        "exercise_name": exercise_name,
        "average_angles": averaged,
        "angle_sequence": all_angles
    }


def get_standard_angle_from_db(reference_video_id: int) -> dict:
    return get_reference_angles_from_db(reference_video_id)["average_angles"]
