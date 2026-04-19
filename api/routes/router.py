from fastapi import APIRouter

from api.routes.v1.confirm import route_confirm
from api.routes.v1.new import route_new
from api.routes.v1.newts import route_newts
from api.routes.v1.projects import route_projects
from api.routes.v1.recognize import route_recognize
from api.routes.v1.species import route_species
from api.routes.v1.territories import route_territories


v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(route_confirm.router)
v1_router.include_router(route_new.router)
v1_router.include_router(route_newts.router)
v1_router.include_router(route_projects.router)
v1_router.include_router(route_recognize.router)
v1_router.include_router(route_species.router)
v1_router.include_router(route_territories.router)
