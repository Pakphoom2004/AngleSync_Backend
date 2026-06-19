import io
import base64
import math
import os

from app.services.motion_analysis import (
    analyze_motion,
    extract_joint_angles,
    validate_exercise_match,
)

from app.services.pose_detection import (
    detect_body_keypoints,
    detect_sample_keypoints,
   
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
from app.repository.reference_repository import (
    get_reference_angles_from_db
)
from app.services.video_validation import (
    verify_file_type,
    verify_video_duration
)

GRAPH_VISIBLE_RATIO = 0.95
MAX_GRAPH_SAMPLES = 180


def _build_graph_samples(
        risk_scores,
        angles_per_frame,
        highest_risk_frame_index
):
    visible_count = max(
        1,
        int(math.ceil(len(risk_scores) * GRAPH_VISIBLE_RATIO))
    )

    if visible_count <= MAX_GRAPH_SAMPLES:
        graph_indices = list(range(visible_count))
    else:
        graph_indices = sorted({
            round(index * (visible_count - 1) / (MAX_GRAPH_SAMPLES - 1))
            for index in range(MAX_GRAPH_SAMPLES)
        })

    if (
        highest_risk_frame_index < visible_count
        and highest_risk_frame_index not in graph_indices
    ):
        graph_indices.append(highest_risk_frame_index)
        graph_indices.sort()

        while len(graph_indices) > MAX_GRAPH_SAMPLES:
            remove_index = min(
                (
                    index
                    for index in graph_indices
                    if index != highest_risk_frame_index
                ),
                key=lambda index: abs(index - highest_risk_frame_index)
            )
            graph_indices.remove(remove_index)

    graph_peak_index = graph_indices.index(highest_risk_frame_index)

    return (
        [risk_scores[index] for index in graph_indices],
        [angles_per_frame[index] for index in graph_indices],
        graph_peak_index
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
        if angles is not None:
            sample_angle_sequence.append(angles)

    if len(sample_angle_sequence) < 5:
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

    # STEP 2: Full detection
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
    (
        graph_risk_scores,
        graph_angles_per_frame,
        graph_peak_index
    ) = _build_graph_samples(
        risk_scores,
        angles_per_frame,
        highest_risk_frame_index
    )

    report_progress(82, "generating_graph", "Generating risk graph...")

    graph_image = generate_risk_graph(
        graph_risk_scores,
        graph_peak_index,
        frame_times=[frame["time"] for frame in graph_angles_per_frame]
    )

    # แปลง PIL Image เป็น base64
    buf = io.BytesIO()
    graph_image.save(buf, format="PNG")
    graph_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    frame_path = save_highest_risk_frame(
        video_path,
        highest_frame_data["frame"],
        highest_frame_data.get("keypoints")
    )
    highest_risk_image_url = (
        f"{base_url}/outputs/{os.path.basename(frame_path)}"
    )

    report_progress(90, "generating_feedback", "Generating feedback...")

    try:
        feedback_result = generate_advanced_feedback(
            highest_frame_data
        )
        print(f"[DEBUG] feedback_result: {feedback_result}")
        feedback = feedback_result["feedback"]
        print(f"[DEBUG] feedback: {feedback}")
    except ServiceException as e:
        print(f"[DEBUG] ServiceException: {e}")
        feedback = {
            "form_summary": "AI feedback unavailable.",
            "injury_risk": str(e),
            "corrective_cues": "Please retry later.",
            "practice_plan": "Please retry later."
        }
    except Exception as e:
        print(f"[DEBUG] Unexpected error: {type(e).__name__}: {e}")
        feedback = {
            "form_summary": "AI feedback unavailable.",
            "injury_risk": str(e),
            "corrective_cues": "Please retry later.",
            "practice_plan": "Please retry later."
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
            "graph_image": f"data:image/png;base64,{graph_base64}",
            "risk_scores": [round(float(s), 2) for s in graph_risk_scores],
            "frame_times": [round(float(f["time"]), 2) for f in graph_angles_per_frame],
            "highest_risk_frame_index": graph_peak_index,
            "highest_risk_image_url": highest_risk_image_url,
        },
        "feedback": feedback
    }
