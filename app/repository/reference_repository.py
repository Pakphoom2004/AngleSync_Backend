
def get_reference_angles_from_db(reference_video_id: int) -> dict:
    from sqlalchemy import bindparam, text

    from app.config.db import get_connection

    with get_connection() as conn:
        reference = conn.execute(
            text(
                """
                SELECT exercise_name
                FROM exercise_reference
                WHERE reference_video_id = :reference_video_id
                LIMIT 1
                """
            ),
            {"reference_video_id": reference_video_id},
        ).mappings().all()

        exercise_name = ""
        if reference:
            exercise_name = reference[0].get("exercise_name") or ""

        # step 1
        frames = conn.execute(
            text(
                """
                SELECT reference_frame_id, frame_sequence
                FROM exercise_reference_frame_data
                WHERE parent_video_id = :reference_video_id
                ORDER BY frame_sequence
                """
            ),
            {"reference_video_id": reference_video_id},
        ).mappings().all()

        print(f"DEBUG reference_video_id: {reference_video_id}")
        print(f"DEBUG frames found: {len(frames)}")

        if not frames:
            raise ValueError(f"No frames found for reference_video_id={reference_video_id}")

        frame_ids = [
            f["reference_frame_id"]
            for f in frames
        ]
        frame_order = {
            f["reference_frame_id"]: index
            for index, f in enumerate(frames)
        }

        # step 2
        metrics_stmt = text(
            """
            SELECT related_frame_id, joint_angle_data
            FROM exercise_reference_pose_metrics
            WHERE related_frame_id IN :frame_ids
            """
        ).bindparams(bindparam("frame_ids", expanding=True))

        metrics = conn.execute(
            metrics_stmt,
            {"frame_ids": frame_ids},
        ).mappings().all()

    print(f"DEBUG metrics found: {len(metrics)}")

    if not metrics:
        raise ValueError(f"No metrics found for frame_ids={frame_ids[:5]}")

    sorted_metrics = sorted(
        metrics,
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
