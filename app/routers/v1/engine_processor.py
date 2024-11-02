from fastapi import (
    APIRouter,
    File,
    UploadFile,
    status,
)
from typing import List

from fastapi.responses import JSONResponse
from app.constants import (
    ENGINE_PROCESSOR_TAGS,
    ENGINE_PROCESSOR_URL, API_V1_PREFIX
)
from app.schemas.predictions import PredictionRequest
from app.services.machine_learning import MachineLearningService
from app.config import settings

import os


router = APIRouter(prefix=API_V1_PREFIX + "/" + ENGINE_PROCESSOR_URL, tags=ENGINE_PROCESSOR_TAGS)


@router.post(
    path="",
    response_description="Process the raw data and train candidates model",
    response_model=None,
    status_code=status.HTTP_201_CREATED,
    response_model_by_alias=False,
)
async def create_engine_processor_model() -> None:

    ml_service = MachineLearningService(settings.FILES_DIRECTORY, training=True)
    ml_service.load_model()

    return JSONResponse(content={"message": "Model trained successfully"})


@router.post(
    path="/{model_name}/files",
    response_description="Creates a report of possible candidates according input files",
    response_model=None,
    status_code=status.HTTP_201_CREATED,
    response_model_by_alias=False,
)
async def create_candidate_report_files(model_name: str, files: List[UploadFile] = File(...)):
    ml_service = MachineLearningService(settings.FILES_DIRECTORY, model_name=model_name)
    cleaned_input_text = await ml_service.process_data_files(files)
    response_dict = ml_service.predict(cleaned_input_text)
    if response_dict is not None:
        return JSONResponse(content=response_dict)
    
    return JSONResponse(content={"error": "Candidate Report generation failed"}, status_code=500)


@router.post(
    path="/{model_name}/text",
    response_description="Preprocess the raw data",
    response_model=None,
    status_code=status.HTTP_201_CREATED,
    response_model_by_alias=False,
)
async def create_candidate_report(model_name:str,prediction: PredictionRequest):

    ml_service = MachineLearningService(settings.FILES_DIRECTORY, model_name=model_name)
    response_dict = ml_service.predict(prediction.selection_criteria)
    if response_dict is not None:
        return JSONResponse(content=response_dict)
    
    return JSONResponse(content={"error": "Candidate Report generation failed"}, status_code=500)

