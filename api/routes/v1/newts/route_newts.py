from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool

from api.dependencies import get_id_service
from api.error import APIError
from api.routes.v1.newts import service

from services.identification_service import IdentificationService


router = APIRouter()


@router.get("/newts/{newt_id}")
async def get_newt_by_id(
    newt_id: str,
    id_service: IdentificationService = Depends(get_id_service)
):
    try:
        return await run_in_threadpool(
            service.get_newt_by_id,
            newt_id,
            id_service
        )
    except APIError as ex:
        raise HTTPException(status_code=ex.status, detail=str(ex))


@router.get("/newts/{newt_id}/cards")
async def get_cards_by_newt_id(
    newt_id: str,
    id_service: IdentificationService = Depends(get_id_service)
):
    try:
        return await run_in_threadpool(
            service.get_cards_by_newt_id,
            newt_id,
            id_service
        )
    except APIError as ex:
        raise HTTPException(status_code=ex.status, detail=str(ex))


@router.patch("/newts/{newt_id}/card")
async def patch_card_by_newt_id(
    request: Request,
    response: Response,
    newt_id: str,
    id_service: IdentificationService = Depends(get_id_service),
):
    params = dict(await request.form())

    try:
        result = await run_in_threadpool(
            service.patch_card_by_newt_id,
            newt_id,
            params,
            id_service,
        )

        response.status_code = 204
        return result
    except APIError as ex:
        raise HTTPException(status_code=ex.status, detail=str(ex))
