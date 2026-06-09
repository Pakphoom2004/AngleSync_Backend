
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
