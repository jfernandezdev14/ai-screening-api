from typing import List
from app.schemas.base import BaseSchema


class CandidateSchema(BaseSchema):
    candidate_name: str
    skills: List[str]
    experience: int
    education: str
    certifications: List[str]
    score: float

