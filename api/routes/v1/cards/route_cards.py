from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from api.dependencies import get_id_service
from api.error import APIError
from api.routes.v1.cards import service

from services.identification_service import IdentificationService


router = APIRouter()
@router.get("/cards/{card_id}/history")
async def get_card_history(
    newt_id: str,
    card_id: str,
    id_service: IdentificationService = Depends(get_id_service)
):
    try:
        return await run_in_threadpool(
            service.get_card_history,
            newt_id,
            card_id,
            id_service
        )
    except APIError as ex:
        raise HTTPException(status_code=ex.status, detail=str(ex))
