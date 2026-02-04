from pydantic import BaseModel
from typing import Optional


class CorrectionRequest(BaseModel):
    input: str
    top_k: int | None = None
    use_retrieval: bool = True


class CorrectionResponse(BaseModel):
    input: str
    reasoning: str
    correction: str
    error_type: Optional[str] = None
