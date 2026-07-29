from fastapi import APIRouter

router = APIRouter()


def get_supabase_client():
    from app.config.supabase_client import supabase

    return supabase


@router.get("/exercises")
def get_exercises():
    response = get_supabase_client().table("exercise_reference") \
        .select("reference_video_id, exercise_name, reference_video_url, reference_gender") \
        .execute()

    return response.data
