from fastapi import APIRouter

from api.routes.v1.confirm import route_confirm
from api.routes.v1.new import route_new
from api.routes.v1.projects import route_projects
from api.routes.v1.recognize import route_recognize


v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(route_confirm.router)
v1_router.include_router(route_new.router)
v1_router.include_router(route_projects.router)
v1_router.include_router(route_recognize.router)
