from services.motion_analysis import (
    analyze_motion
)

from services.pose_detection import (
    detect_body_keypoints
)

from services.graph_service import (
    generate_risk_graph,
    save_highest_risk_frame
)

from services.feedback_service import (
    generate_advanced_feedback
)

from exceptions import (
    ServiceException
)


def process_video_analysis(
        video_path: str
):

    keypoints_per_frame = (
        detect_body_keypoints(
            video_path
        )
    )

    standard_angle = {
        "left_elbow": 160,
        "right_elbow": 160,
        "left_shoulder": 40,
        "right_shoulder": 40,
        "left_knee": 90,
        "right_knee": 90,
        "left_hip": 100,
        "right_hip": 100
    }

    analysis_result = (
        analyze_motion(
            keypoints_per_frame,
            standard_angle
        )
    )

    accuracy_score = (
        analysis_result[
            "accuracy_score"
        ]
    )

    risk_scores = (
        analysis_result[
            "risk_scores"
        ]
    )

    highest_risk_frame_index = (
        analysis_result[
            "highest_risk_frame_index"
        ]
    )

    angles_per_frame = (
        analysis_result[
            "angles_per_frame"
        ]
    )

    highest_frame_data = (
        angles_per_frame[
            highest_risk_frame_index
        ]
    )

    graph_data = (
        generate_risk_graph(
            risk_scores,
            highest_risk_frame_index
        )
    )

    frame_path = (
        save_highest_risk_frame(
            video_path,
            highest_risk_frame_index
        )
    )

    try:

        feedback_result = (
            generate_advanced_feedback(
                highest_frame_data,
                frame_path
            )
        )


    except ServiceException:

        feedback_result = {
            "form_summary": "AI feedback unavailable.",
            "injury_risk": "Gemini API quota exceeded.",
            "corrective_cues": "Please retry later.",
            "practice_plan": "Analyze again after quota reset."
        }

    if accuracy_score >= 80:
        risk_level = "GOOD"

    elif accuracy_score >= 60:
        risk_level = "NORMAL"

    else:
        risk_level = "DANGEROUS"

    return {
        "score": round(accuracy_score, 2),
        "risk_level": risk_level,
        "selected_frame": {
            "frame": highest_risk_frame_index,
            "time": round(highest_frame_data["time"], 2),
            "risk": round(highest_frame_data["risk_score"], 2)
        },
        "graph_data": graph_data,
        "feedback": feedback_result
    }
