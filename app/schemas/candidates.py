from app.schemas.base import BaseSchema


class CandidateSchema(BaseSchema):
    candidate_name: str
    token_type: str
