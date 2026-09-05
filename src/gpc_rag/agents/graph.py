"""Grafo de agentes (LangGraph): coordinador -> RAG -> evaluador -> coordinador.

    START
      |
      v
coordinator_filter --(no adecuada)-------------------+
      |                                               |
   (adecuada)                                         |
      v                                               v
     rag  <----------------------+          coordinator_finalize --> END
      |                          | (reintento,           ^
      v                          |  maximo 1 vez)         |
  evaluator ---------------------+                        |
      |                                                    |
  (aprobada, o ya no quedan reintentos) --------------------+

El coordinador es el UNICO agente que decide que ve el usuario final
(agents/coordinator.py: build_final_output) -- ni el agente RAG ni el
evaluador devuelven nada directamente.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph
from omegaconf import DictConfig

from gpc_rag.agents.coordinator import build_final_output, filter_query
from gpc_rag.agents.evaluator import evaluate_answer
from gpc_rag.agents.state import AgentState
from gpc_rag.common.models import RagAnswer
from gpc_rag.pipelines.query_pipeline import QueryPipeline

logger = logging.getLogger(__name__)

MAX_RETRIES = 1


def _coordinator_filter_node(state: AgentState, cfg: DictConfig) -> dict:
    is_ok, reason = filter_query(state["question"], cfg)
    logger.info("Coordinador (filtro): pregunta=%r adecuada=%s", state["question"][:60], is_ok)
    return {"is_appropriate": is_ok, "rejection_reason": reason}


def _rag_node(state: AgentState, pipeline: QueryPipeline) -> dict:
    result = pipeline.answer(state["question"], extra_instruction=state.get("extra_instruction"))
    return {"answer": result.answer, "sources": result.sources}


def _evaluator_node(state: AgentState, cfg: DictConfig) -> dict:
    ok, feedback = evaluate_answer(state["question"], state.get("sources", []), state["answer"], cfg)
    logger.info(
        "Evaluador: ok=%s feedback=%r retry_count=%d", ok, feedback, state.get("retry_count", 0)
    )
    return {"eval_ok": ok, "eval_feedback": feedback}


def _prepare_retry_node(state: AgentState) -> dict:
    feedback = state.get("eval_feedback") or "la respuesta no estaba bien fundamentada."
    instruction = (
        f"Tu respuesta anterior fue rechazada por este motivo: {feedback} "
        "Vuelve a responder siendo mas literal: usa solo texto que aparezca "
        "en el CONTEXTO, y si no hay informacion suficiente, dilo "
        "explicitamente en vez de completar con conocimiento propio."
    )
    return {"extra_instruction": instruction, "retry_count": state.get("retry_count", 0) + 1}


def _coordinator_finalize_node(state: AgentState) -> dict:
    final_answer, final_sources = build_final_output(state)
    return {"final_answer": final_answer, "final_sources": final_sources}


def _route_after_filter(state: AgentState) -> str:
    return "rag" if state.get("is_appropriate") else "finalize"


def _route_after_evaluator(state: AgentState) -> str:
    if state.get("eval_ok"):
        return "finalize"
    if state.get("retry_count", 0) < MAX_RETRIES:
        return "retry"
    return "finalize"


def build_agent_graph(cfg: DictConfig, pipeline: QueryPipeline):
    graph = StateGraph(AgentState)

    graph.add_node("coordinator_filter", lambda s: _coordinator_filter_node(s, cfg))
    graph.add_node("rag", lambda s: _rag_node(s, pipeline))
    graph.add_node("evaluator", lambda s: _evaluator_node(s, cfg))
    graph.add_node("prepare_retry", _prepare_retry_node)
    graph.add_node("coordinator_finalize", _coordinator_finalize_node)

    graph.set_entry_point("coordinator_filter")
    graph.add_conditional_edges(
        "coordinator_filter",
        _route_after_filter,
        {"rag": "rag", "finalize": "coordinator_finalize"},
    )
    graph.add_edge("rag", "evaluator")
    graph.add_conditional_edges(
        "evaluator",
        _route_after_evaluator,
        {"finalize": "coordinator_finalize", "retry": "prepare_retry"},
    )
    graph.add_edge("prepare_retry", "rag")
    graph.add_edge("coordinator_finalize", END)

    return graph.compile()


def run_agentic(question: str, cfg: DictConfig, pipeline: QueryPipeline) -> RagAnswer:
    """Punto de entrada usado por la API (ver api/main.py, flag GPC_USE_AGENTS)."""
    app = build_agent_graph(cfg, pipeline)
    result_state = app.invoke({"question": question, "retry_count": 0})
    return RagAnswer(
        question=question,
        answer=result_state["final_answer"],
        sources=result_state.get("final_sources", []),
    )
