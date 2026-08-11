import logging
from typing import Any, Dict, List

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import asyncio
import json
import os
import queue
import shutil
import threading

from app.exceptions.save_transaction_failed_exception import SaveTransactionFailedException
from app.services.analysis_results import process_video_analysis
from app.services.save_analyze import save_analysis_result, logger

router = APIRouter()
UPLOAD_DIR = "uploads"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

class SaveAnalyzeRequest(BaseModel):
    user_id: int
    session_name: str
    reference_video_id: int
    video_user_url: str
    accuracy_score: float
    risk_frames: List[Dict[str, Any]] = Field(default_factory=list)
    feedback: Dict[str, Any]


def get_supabase_client():
    from app.config.supabase_client import supabase

    return supabase


@router.post("/analyze/stream")
async def analyze_video_stream(
    request: Request,
    file: UploadFile = File(...),
    reference_video_id: int = Form(...)
):
    base_url = str(request.base_url).rstrip("/")

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
                    progress_callback=progress_callback,
                    
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


@router.post("/save-analyze")
async def save_analyze(payload: SaveAnalyzeRequest):
    payload_data = (
        payload.model_dump()
        if hasattr(payload, "model_dump")
        else payload.dict()
    )

    logger.info(f"save-analyze risk_frames payload: {payload_data.get('risk_frames')}")

    try:
        return save_analysis_result(
            get_supabase_client(),
            **payload_data
        )
    except SaveTransactionFailedException as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
