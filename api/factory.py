from fastapi import FastAPI
from contextlib import asynccontextmanager

from api.routes import router
from api.services.temp import TempStorage
from services.identification_service import create_identification_service


@asynccontextmanager
async def lifespan(api: FastAPI):
    service = create_identification_service()

    temp = TempStorage()

    api.state.id_service = service
    api.state.card_service = service.card_service
    api.state.temp = temp

    yield

    # cleanup on shutdown
    temp.cleanup()


def make_app() -> FastAPI:
    app = FastAPI(
        title="Newt Identification by Tutochki",
        lifespan=lifespan
    )

    app.include_router(router.v1_router)

    return app
