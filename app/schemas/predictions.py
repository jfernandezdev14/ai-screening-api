from typing import List

from fastapi import File, UploadFile
from app.schemas.base import BaseSchema


class PredictionRequest(BaseSchema):
    selection_criteria: str


class PredictionFileRequest(BaseSchema):
    files: List[UploadFile] = File(...)




