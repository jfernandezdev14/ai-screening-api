from typing import List
from app.schemas.base import BaseSchema


class PredictionRequest(BaseSchema):
    selection_criteria: str

