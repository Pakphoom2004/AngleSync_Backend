from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile
)
import shutil
import os

from exceptions import (
    InvalidKeypointsException,
    KeypointNotDetectedException,
    ServiceException
)

from services.analysis_results import (
    process_video_analysis
)

router = APIRouter()
UPLOAD_DIR = "uploads"

@router.post("/analyze")
async def analyze_video(
    video: UploadFile = File(...)
):

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    file_path = f"{UPLOAD_DIR}/{video.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    try:
        result = process_video_analysis(file_path)
    except (
        KeypointNotDetectedException,
        InvalidKeypointsException
    ) as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        ) from error
    except ServiceException as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        ) from error

    return result
