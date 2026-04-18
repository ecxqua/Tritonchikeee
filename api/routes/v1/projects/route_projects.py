from fastapi import APIRouter, Depends, Form, HTTPException, Response
from starlette.status import HTTP_201_CREATED
from fastapi.concurrency import run_in_threadpool

from api.dependencies import get_id_service
from api.error import APIError
from api.routes.v1.projects import service


router = APIRouter()


@router.post("/projects")
async def create_project(
    response: Response,
    name: str = Form(...),
    description: str = Form(...),
    species: str | None = Form(...),
    territory: str | None = Form(...),
    id_service=Depends(get_id_service),
):
    try:
        result = await run_in_threadpool(
            service.create_project,
            name,
            description,
            species,
            territory,
            id_service,
        )

        response.status_code = HTTP_201_CREATED
        return result
    except APIError as ex:
        raise HTTPException(status_code=ex.status, detail=str(ex))


@router.get("/projects")
async def fetch_projects(
    id_service=Depends(get_id_service),
):
    try:
        return await run_in_threadpool(
            service.fetch_projects,
            id_service,
        )
    except APIError as ex:
        raise HTTPException(status_code=ex.status, detail=str(ex))