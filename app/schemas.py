from pydantic import BaseModel, Field
from typing import Optional


class SummarizeRequest(BaseModel):
text: str = Field(..., min_length=20)
style: str = Field("formal", pattern=r"^(formal|casual|bullet)$")
max_tokens: int = 200
temperature: float = 0.2


class SummarizeResponse(BaseModel):
style: str
summary: str
