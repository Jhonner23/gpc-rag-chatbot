"""API FastAPI que expone el RAG: recibe una pregunta, devuelve respuesta + fuentes.

Correr local:
    uv run uvicorn gpc_rag.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.memory import MemorySaver

from gpc_rag.agents.graph import run_agentic
from gpc_rag.api.schemas import ChatRequest, ChatResponse, SourceOut
from gpc_rag.common.models import PendingTreeQuestion
from gpc_rag.pipelines.query_pipeline import get_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bandera para activar la capa de agentes (coordinador + arbol + RAG +
# evaluador + comparador, ver src/gpc_rag/agents/) en vez del pipeline
# directo. Detras de un flag para poder comparar el comportamiento y la
# latencia de ambos caminos durante la validacion, sin arriesgar el
# pipeline ya probado en produccion.
USE_AGENTS = os.environ.get("GPC_USE_AGENTS", "false").lower() in {"1", "true", "yes"}

# Un solo checkpointer en memoria, compartido por todas las peticiones de
# este proceso: es lo que permite que el agente arbol pause el grafo
# (interrupt(), ver agents/tree_agent.py) en una peticion HTTP y lo
# reanude en la siguiente, identificando la conversacion por thread_id. Se
# pierde si el proceso se reinicia -- aceptable para esta tesis (un solo
# proceso de API), pero no serviria para un despliegue con varias replicas
# sin un checkpointer compartido (p.ej. en Postgres/Redis).
_tree_checkpointer = MemorySaver()

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
    is_resume = bool(request.thread_id and request.answer is not None)

    if not is_resume:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="La pregunta no puede estar vacia.")

    if is_resume and not USE_AGENTS:
        raise HTTPException(
            status_code=400,
            detail="El flujo de preguntas del agente arbol requiere GPC_USE_AGENTS=true.",
        )

    try:
        pipeline = get_pipeline()
        if not USE_AGENTS:
            result = pipeline.answer(question)
        elif is_resume:
            result = run_agentic(
                "",
                pipeline.cfg,
                pipeline,
                checkpointer=_tree_checkpointer,
                thread_id=request.thread_id,
                resume_answer=request.answer,
            )
        else:
            result = run_agentic(question, pipeline.cfg, pipeline, checkpointer=_tree_checkpointer)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Fallo el pipeline de RAG para la peticion: %s", request)
        raise HTTPException(
            status_code=503,
            detail="El servicio de RAG no esta disponible en este momento (revisa Ollama/Qdrant).",
        ) from exc

    if isinstance(result, PendingTreeQuestion):
        return ChatResponse(
            question=request.question,
            answer=None,
            sources=[],
            pending_question=result.question,
            thread_id=result.thread_id,
        )

    return ChatResponse(
        question=result.question,
        answer=result.answer,
        sources=[SourceOut(**s) for s in result.sources_as_dicts()],
    )
