from fastapi import (
    APIRouter,
    File,
    UploadFile,
    status,
)
from typing import List

from app.config import settings
from app.constants import (
    CANDIDATES_TAGS,
    CANDIDATES_URL, API_V1_PREFIX
)
from app.services.machine_learning import MachineLearningService

import os


router = APIRouter(prefix=API_V1_PREFIX + "/" + CANDIDATES_URL, tags=CANDIDATES_TAGS)


@router.post(
    path="/process",
    response_description="Preprocess the raw data",
    response_model=None,
    status_code=status.HTTP_201_CREATED,
    response_model_by_alias=False,
)
async def process_candidates() -> None:

    ml_service = MachineLearningService("./db/data.csv")
    ml_service.load("./db/data.csv")

    return True


@router.post(
    path="",
    response_description="Create new candidates and process their information",
    response_model=None,
    status_code=status.HTTP_201_CREATED,
    response_model_by_alias=False,
)
async def create_candidate(files: List[UploadFile] = File(...)):

    # Save the uploaded file
    file_location = os.path.join(settings.FILES_DIRECTORY, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    return {"info": f"File '{file.filename}' has been uploaded successfully."}


@router.get(
    path="",
    response_description="Preprocess the raw data",
    response_model=None,
    status_code=status.HTTP_201_CREATED,
    response_model_by_alias=False,
)
async def find_candidates() -> None:

    ml_service = MachineLearningService("./db/data.csv")
    ml
    ml_service.load("./db/data.csv")

    return True

