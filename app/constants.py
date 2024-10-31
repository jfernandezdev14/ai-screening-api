from enum import Enum
from typing import (
    Final, List,
)

# Open API parameters
AI_SCREENING_API_TITLE: Final = "API Hub"
AI_SCREENING_API_DESCRIPTION: Final = "AI-Powered Candidate Screening and Scoring System."

# API URLS
API_V1_PREFIX: Final = "/api/v1"

# DATA_PROCESSING service constants
CANDIDATES_TAGS: Final[List[str | Enum] | None] = ["Candidates API"]
CANDIDATES_URL: Final = "candidates"

