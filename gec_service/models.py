from pydantic import BaseModel
from typing import Optional


class CorrectionRequest(BaseModel):
    input: str


class CorrectionResponse(BaseModel):
    input: str
    reasoning: str
    correction: str
    error_type: Optional[str] = None
