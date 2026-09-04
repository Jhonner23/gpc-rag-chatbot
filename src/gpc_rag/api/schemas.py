from __future__ import annotations

from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class SourceOut(BaseModel):
    source_file: str
    section: str
    page: int | None = None
    score: float


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceOut]
