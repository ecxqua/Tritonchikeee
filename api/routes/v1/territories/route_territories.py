from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from api.dependencies import get_id_service
from api.error import APIError
from api.routes.v1.territories import service

from services.identification_service import IdentificationService


router = APIRouter()


@router.get("/territories")
async def fetch_territories(
    id_service: IdentificationService = Depends(get_id_service),
):
    try:
        return await run_in_threadpool(
            service.fetch_territories,
            id_service,
        )
    except APIError as ex:
        raise HTTPException(status_code=ex.status, detail=str(ex))
