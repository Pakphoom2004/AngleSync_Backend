import json
from datetime import datetime

# Save complete analysis session locally
def save_session_data(
        session_name: str,
        accuracy_score: float,
        risk_graph: str,
        skeleton_overlay: str,
        feedback: dict,
        joint_coordinates: tuple,
        user_video_id: int,
        reference_id: int
):

    analysis_date = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    session_data = {
        "session_name": session_name,
        "analysis_date": analysis_date,
        "accuracy_score": accuracy_score,
        "risk_graph": risk_graph,
        "skeleton_overlay": skeleton_overlay,
        "feedback": feedback,
        "joint_coordinates": joint_coordinates,
        "user_video_id": user_video_id,
        "reference_id": reference_id
    }

    output_path = (
        f"outputs/{session_name}.json"
    )

    with open(output_path, "w") as file:
        json.dump(
            session_data,
            file,
            indent=4
        )

    return True