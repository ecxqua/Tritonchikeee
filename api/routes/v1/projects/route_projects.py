from fastapi import APIRouter, Body, Depends, Form, HTTPException, Response
from fastapi.concurrency import run_in_threadpool

from api.dependencies import get_id_service
from api.error import APIError
from api.routes.v1.projects import service

from services.identification_service import IdentificationService

from typing import List


router = APIRouter()


@router.post("/projects")
async def create_project(
    response: Response,
    name: str = Form(...),
    description: str = Form(...),
    species: List[str] | None = Form(None),
    territory: List[str] | None = Form(None),
    id_service: IdentificationService = Depends(get_id_service),
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

        response.status_code = 201
        return result
    except APIError as ex:
        raise HTTPException(status_code=ex.status, detail=str(ex))


@router.get("/projects")
async def fetch_projects(
    id_service: IdentificationService = Depends(get_id_service),
):
    try:
        return await run_in_threadpool(
            service.fetch_projects,
            id_service,
        )
    except APIError as ex:
        raise HTTPException(status_code=ex.status, detail=str(ex))


@router.get("/projects/{project_id}")
async def fetch_project(
    project_id: int,
    id_service: IdentificationService = Depends(get_id_service)
):
    try:
        return await run_in_threadpool(
            service.fetch_project,
            project_id,
            id_service,
        )
    except APIError as ex:
        raise HTTPException(status_code=ex.status, detail=str(ex))


@router.patch("/projects/{project_id}")
async def patch_project(
    response: Response,
    project_id: int,
    name: str | None = Body(None),
    description: str | None = Body(None),
    species: List[str] | None = Body(None),
    territory: List[str] | None = Body(None),
    id_service: IdentificationService = Depends(get_id_service),
):
    try:
        result = await run_in_threadpool(
            service.update_project,
            project_id,
            name,
            description,
            species,
            territory,
            id_service,
        )

        response.status_code = 204
        return result
    except APIError as ex:
        raise HTTPException(status_code=ex.status, detail=str(ex))


@router.delete("/projects/{project_id}")
async def delete_project(
    response: Response,
    project_id: int,
    id_service: IdentificationService = Depends(get_id_service)
):
    try:
        result = await run_in_threadpool(
            service.delete_project,
            project_id,
            id_service,
        )

        response.status_code = 204
        return result
    except APIError as ex:
        raise HTTPException(status_code=ex.status, detail=str(ex))


@router.get("/projects/{project_id}/newts")
async def get_project_newts(
    project_id: int,
    id_service: IdentificationService = Depends(get_id_service)
):
    try:
        return await run_in_threadpool(
            service.get_project_newts,
            project_id,
            id_service
        )
    except APIError as ex:
        raise HTTPException(status_code=ex.status, detail=str(ex))
