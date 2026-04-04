from fastapi import APIRouter

from api.routes.v1.analyse import route_analyse
from api.routes.v1.confirm import route_confirm


v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(route_analyse.router)
v1_router.include_router(route_confirm.router)
