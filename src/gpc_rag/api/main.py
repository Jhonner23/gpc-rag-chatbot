"""API FastAPI que expone el RAG: recibe una pregunta, devuelve respuesta + fuentes.

Correr local:
    uv run uvicorn gpc_rag.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from gpc_rag.api.schemas import ChatRequest, ChatResponse, SourceOut
from gpc_rag.pipelines.query_pipeline import get_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GPC RAG Chatbot API",
    description="Consulta Guias de Practica Clinica via RAG (on-premise, sin dependencias de pago).",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacia.")

    try:
        pipeline = get_pipeline()
        result = pipeline.answer(question)
    except Exception as exc:
        logger.exception("Fallo el pipeline de RAG para la pregunta: %s", question)
        raise HTTPException(
            status_code=503,
            detail="El servicio de RAG no esta disponible en este momento (revisa Ollama/Qdrant).",
        ) from exc

    return ChatResponse(
        question=result.question,
        answer=result.answer,
        sources=[SourceOut(**s) for s in result.sources_as_dicts()],
    )
