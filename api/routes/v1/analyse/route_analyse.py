from fastapi import APIRouter, UploadFile, File, Depends
from fastapi.concurrency import run_in_threadpool

from api.dependencies import get_id_service, get_temp
from api.routes.v1.analyse import service
from api.models.file_data import FileData

from pathlib import Path


router = APIRouter()


@router.post("/analyse/")
async def analyse(
    file: UploadFile = File(...),  # multipart/form-data request,
    id_service = Depends(get_id_service),
    temp = Depends(get_temp)
):
    if file.content_type not in ["image/png", "image/jpeg"]:
        return {"error": "Invalid file type"}
    
    contents = await file.read()
    original_name = Path(file.filename)

    file_data = FileData(
        name=original_name.stem,
        ext=original_name.suffix,
        data=contents
    )

    # async ML work is pointless, it's still CPU-bound and blocking
    # use a threadpool and multiple uvicorn workers instead
    # e.g. `uvicorn <entrypoint> --workers 4`
    return await run_in_threadpool(
        service.complete_analyse,
        file_data,
        id_service,
        temp
    )