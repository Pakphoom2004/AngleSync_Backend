from fastapi import APIRouter
from app.config.supabase_client import supabase

router = APIRouter()

@router.get("/exercises")
def get_exercises():
    response = supabase.table("exercise_reference") \
        .select("reference_video_id, exercise_name, reference_video_url, reference_gender") \
        .execute()

    return response.data