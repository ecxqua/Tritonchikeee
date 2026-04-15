from fastapi import APIRouter, Depends, Form, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from api.dependencies import get_id_service, get_card_service, get_temp
from api.models import FileData
from api.routes.v1.new import service
from api.error import APIError

from pathlib import Path


router = APIRouter()


@router.post("/new")
async def new(
    request: Request,
    file: UploadFile = File(...),  # multipart/form-data request
    species: str = Form(...),
    project_id: str | None = Form(...),
    template_type: str = Form(...),
    card_id: str | None = Form(...),
    id_service=Depends(get_id_service),
    card_service=Depends(get_card_service),
    temp=Depends(get_temp)
):
    contents = await file.read()
    original_name = Path(file.filename)

    file_data = FileData(
        name=original_name.stem,
        ext=original_name.suffix,
        data=contents
    )

    params = dict(await request.form())
    str_params = {k: v for k, v in params.items() if isinstance(v, str)}

    try:
        return await run_in_threadpool(
            service.add_new_card,
            file_data,
            species,
            project_id,
            template_type,
            card_id,
            str_params,
            id_service,
            card_service,
            temp
        )
    except APIError as ex:
        raise HTTPException(status_code=ex.status, detail=str(ex))
