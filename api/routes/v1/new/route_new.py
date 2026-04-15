from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from api.dependencies import get_id_service
from api.routes.v1.confirm import service
from api.error import APIError


router = APIRouter()


@router.post("/new")
async def new(
    request: Request,
    species: str = Form(...),
    project_id: str | None = Form(...),
    template_type: str = Form(...),
    card_id: str = Form(...),
    
    id_service=Depends(get_id_service)
):
    params = dict(await request.form())

    try:
        return await run_in_threadpool(
            service.complete_confirmation,
            upload_id,
            decision,
            existing_id,
            params,
            id_service
        )
    except APIError as ex:
        raise HTTPException(status_code=ex.status, detail=str(ex))
