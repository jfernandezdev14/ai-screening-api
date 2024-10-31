# from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.constants import (
    AI_SCREENING_API_DESCRIPTION,
    AI_SCREENING_API_TITLE,
)
from app.routers.v1 import candidates
from app.utils.exceptions.exceptions import IException

from app.version import __version__

# Load the .env file
load_dotenv()

app = FastAPI(
    title=AI_SCREENING_API_TITLE,
    description=AI_SCREENING_API_DESCRIPTION,
    version=__version__,
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
)


# @app.on_event("startup")
# async def startup_db_client():
#     app.mongodb_client = AsyncIOMotorClient(settings.DB_URL, tls=True, tlsAllowInvalidCertificates=True)
#     app.mongodb = app.mongodb_client[settings.DB_NAME]


# @app.on_event("shutdown")
# async def shutdown_db_client():
#     app.mongodb_client.close()


@app.exception_handler(IException)
async def unicorn_exception_handler(request: Request, exc: IException):
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.content,
    )


@app.get("/api/health_check")
def health_check():
    return "ok"


app.include_router(candidates.router)