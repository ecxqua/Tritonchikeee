from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from api.dependencies import get_id_service, get_temp
from api.routes.v1.confirm import service
from api.error import APIError


router = APIRouter()


@router.post("/confirm")
async def confirm(
    request: Request,
    upload_id: int = Form(...),
    decision: str = Form(...),
    existing_id: str | None = Form(...),
    id_service = Depends(get_id_service),
    temp = Depends(get_temp)
):
    params = dict(await request.form())
    
    try:
        return await run_in_threadpool(
            service.complete_confirmation,
            upload_id,
            decision,
            existing_id,
            params,
            id_service,
            temp
        )
    except APIError as ex:
        raise HTTPException(status_code=ex.status, detail=str(ex))