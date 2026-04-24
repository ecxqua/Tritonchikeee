from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.routes import router
from api.services.temp import TempStorage
from services.identification_service import create_identification_service


@asynccontextmanager
async def lifespan(api: FastAPI):
    service = create_identification_service()

    temp = TempStorage()

    api.state.id_service = service
    api.state.temp = temp

    yield

    # cleanup on shutdown
    temp.cleanup()


def make_app() -> FastAPI:
    app = FastAPI(
        title="Newt Identification by Tutochki",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router.v1_router)

    return app
