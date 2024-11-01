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
ENGINE_PROCESSOR_TAGS: Final[List[str | Enum] | None] = ["Engine Processor API"]
ENGINE_PROCESSOR_URL: Final = "engine"

