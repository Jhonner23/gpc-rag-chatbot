"""Estado compartido entre los agentes del grafo (coordinador, RAG, evaluador).

Ver agents/graph.py para el flujo completo. Se usa TypedDict (no dataclass)
porque es lo que LangGraph espera como esquema de estado de un StateGraph.
"""

from __future__ import annotations

from typing import TypedDict

from gpc_rag.common.models import RetrievedChunk


class AgentState(TypedDict, total=False):
    question: str

    # Agente coordinador (pre-filtro de admision)
    is_appropriate: bool
    rejection_reason: str | None

    # Agente RAG (retrieval + generacion)
    answer: str
    sources: list[RetrievedChunk]
    extra_instruction: str | None  # se llena si el evaluador pide reintentar

    # Agente evaluador (guardrail en linea, antes de responder al usuario)
    eval_ok: bool
    eval_feedback: str | None
    retry_count: int

    # Salida final: la arma el coordinador (unico agente que "habla" con el
    # usuario), decide si se muestran fuentes o no segun el resultado.
    final_answer: str
    final_sources: list[RetrievedChunk]
