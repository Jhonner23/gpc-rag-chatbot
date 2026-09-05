"""API FastAPI que expone el RAG: recibe una pregunta, devuelve respuesta + fuentes.

Correr local:
    uv run uvicorn gpc_rag.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException

from gpc_rag.agents.graph import run_agentic
from gpc_rag.api.schemas import ChatRequest, ChatResponse, SourceOut
from gpc_rag.pipelines.query_pipeline import get_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bandera para activar la capa de agentes (coordinador + RAG + evaluador, ver
# src/gpc_rag/agents/) en vez del pipeline directo. Detras de un flag para
# poder comparar el comportamiento y la latencia de ambos caminos durante la
# validacion, sin arriesgar el pipeline ya probado en produccion.
USE_AGENTS = os.environ.get("GPC_USE_AGENTS", "false").lower() in {"1", "true", "yes"}

app = FastAPI(
    title="GPC RAG Chatbot API",
    description="Consulta Guias de Practica Clinica via RAG (on-premise, sin dependencias de pago).",
    version="0.1.0",
)


@app.on_event("startup")
def _warm_up_pipeline() -> None:
    try:
        pipeline = get_pipeline()
        if pipeline.retriever.cfg.retrieval.use_reranker:
            _ = pipeline.retriever.reranker
        logger.info("Pipeline RAG precargado.")
    except Exception:
        logger.exception("No se pudo precargar el pipeline en el startup.")


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
        if USE_AGENTS:
            result = run_agentic(question, pipeline.cfg, pipeline)
        else:
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
