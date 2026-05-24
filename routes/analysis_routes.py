from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
import json
import os
import shutil

from services.analysis_results import process_video_analysis

router = APIRouter()
UPLOAD_DIR = "uploads"


@router.post("/analyze/stream")
async def analyze_video_stream(
    file: UploadFile = File(...)
):

    async def event_generator():
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        file_path = f"{UPLOAD_DIR}/{file.filename}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # STEP 1
        payload = json.dumps({
            "step": "uploading",
            "message": "Uploading video...",
            "percent": 10
        })
        yield f"event: step\ndata: {payload}\n\n"

        # STEP 2
        payload = json.dumps({
            "step": "detecting",
            "message": "Detecting pose...",
            "percent": 30
        })
        yield f"event: progress\ndata: {payload}\n\n"

        result = process_video_analysis(file_path)

        # STEP 3
        payload = json.dumps({
            "step": "analyzing",
            "message": "Analyzing movement...",
            "percent": 70
        })
        yield f"event: progress\ndata: {payload}\n\n"

        # FINAL RESULT
        payload = json.dumps(result)
        yield f"event: result\ndata: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )