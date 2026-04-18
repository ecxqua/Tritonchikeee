from fastapi import APIRouter, UploadFile, File, Depends, Form, HTTPException
from fastapi.concurrency import run_in_threadpool
from api.error import APIError

from api.dependencies import get_id_service, get_temp
from api.routes.v1.recognize import service
from api.models import FileData

from pathlib import Path


router = APIRouter()


@router.post("/recognize")
async def recognize(
    photo: UploadFile = File(...),  # multipart/form-data request
    scope: str = Form(...),
    projectId: int | None = Form(...),
    id_service=Depends(get_id_service),
    temp=Depends(get_temp)
):
    if photo.content_type not in ["image/png", "image/jpeg"]:
        return {"error": "Invalid file type"}

    contents = await photo.read()
    original_name = Path(photo.filename or "...")

    file_data = FileData(
        name=original_name.stem,
        ext=original_name.suffix,
        data=contents
    )

    # async ML work is pointless, it's still CPU-bound and blocking
    # use a threadpool and multiple uvicorn workers instead
    # e.g. `uvicorn <entrypoint> --workers 4`
    try:
        return await run_in_threadpool(
            service.complete_recognize,
            file_data,
            scope,
            projectId,
            id_service,
            temp
        )
    except APIError as ex:
        raise HTTPException(status_code=ex.status, detail=str(ex))
