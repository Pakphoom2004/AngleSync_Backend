from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
import asyncio
import json
import os
import queue
import shutil
import threading

from app.services.analysis_results import process_video_analysis
from fastapi import Form

router = APIRouter()
UPLOAD_DIR = "uploads"


@router.post("/analyze/stream")
async def analyze_video_stream(
    file: UploadFile = File(...),
    reference_video_id: int = Form(...)
):

    async def event_generator():
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        events = queue.Queue()

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

        def progress_callback(progress_payload):
            events.put(
                (
                    "progress",
                    progress_payload
                )
            )

        def run_analysis():
            try:
                result = process_video_analysis(
                    file_path,
                    reference_video_id,
                    progress_callback=progress_callback
                )
                events.put(
                    (
                        "progress",
                        {
                            "step": "completed",
                            "message": "Analysis completed.",
                            "percent": 100
                        }
                    )
                )
                events.put(
                    (
                        "result",
                        result
                    )
                )

            except Exception as error:
                events.put(
                    (
                        "error",
                        {
                            "status": "error",
                            "message": str(error),
                            "percent": 100
                        }
                    )
                )

        worker = threading.Thread(
            target=run_analysis,
            daemon=True
        )
        worker.start()

        while True:
            event_name, event_payload = await asyncio.to_thread(
                events.get
            )
            payload = json.dumps(event_payload)
            yield f"event: {event_name}\ndata: {payload}\n\n"

            if event_name in {
                "result",
                "error"
            }:
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
