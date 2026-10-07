"""Grafo de agentes (LangGraph).

    START
      |
      v
coordinator_filter --(no adecuada)----------------------------------+
      |                                                              |
   (adecuada)                                                        |
      v                                                              |
  tree_prepare  (decide protocolo NAC/ITU/ninguno + 1ra extraccion)  |
      |                                                              |
      v                                                              |
  tree_agent  (motor cpg_tree; interrupt() -> wizard si faltan datos)|
      |                                                              |
      v                                                              |
     rag  <----------------------+                                  |
      |                          | (reintento,                      |
      v                          |  maximo 1 vez)                    |
  evaluator ---------------------+                                  |
      |                                                              |
  (aprobada, o ya no quedan reintentos)                              |
      |                                                              |
      v                                                              |
  comparador  (arbol auditado si hizo match, si no el RAG evaluado)  |
      |                                                              |
      v                                                              v
                      coordinator_finalize --> END

El coordinador es el UNICO agente que decide que ve el usuario final
(agents/coordinator.py: build_final_output) -- ni el agente RAG, el
evaluador, el agente arbol, ni el comparador devuelven nada directamente.

El agente arbol puede pausar el grafo a mitad de camino (LangGraph
`interrupt()`) para preguntarle al usuario, una variable a la vez, estilo
wizard -- ver agents/tree_agent.py. Esa pausa/resume se identifica con un
`thread_id` que persiste en el checkpointer (ver run_agentic): la API
(api/main.py) se lo devuelve al chat, y el chat se lo reenvia junto con la
respuesta a la pregunta pendiente.
"""

from __future__ import annotations

import logging
import uuid

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from omegaconf import DictConfig

from gpc_rag.agents.comparator import compare_and_choose
from gpc_rag.agents.coordinator import build_final_output, filter_query
from gpc_rag.agents.evaluator import evaluate_answer
from gpc_rag.agents.state import AgentState
from gpc_rag.agents.tree_agent import tree_agent_node, tree_prepare_node
from gpc_rag.common.models import PendingTreeQuestion, RagAnswer
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
    ok, feedback = evaluate_answer(
        state["question"], state.get("sources", []), state["answer"], cfg
    )
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


def build_agent_graph(
    cfg: DictConfig,
    pipeline: QueryPipeline,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("coordinator_filter", lambda s: _coordinator_filter_node(s, cfg))
    graph.add_node("tree_prepare", lambda s: tree_prepare_node(s, cfg))
    graph.add_node("tree_agent", lambda s: tree_agent_node(s, cfg))
    graph.add_node("rag", lambda s: _rag_node(s, pipeline))
    graph.add_node("evaluator", lambda s: _evaluator_node(s, cfg))
    graph.add_node("prepare_retry", _prepare_retry_node)
    graph.add_node("comparador", compare_and_choose)
    graph.add_node("coordinator_finalize", _coordinator_finalize_node)

    graph.set_entry_point("coordinator_filter")
    graph.add_conditional_edges(
        "coordinator_filter",
        _route_after_filter,
        {"rag": "tree_prepare", "finalize": "coordinator_finalize"},
    )
    graph.add_edge("tree_prepare", "tree_agent")
    graph.add_edge("tree_agent", "rag")
    graph.add_edge("rag", "evaluator")
    graph.add_conditional_edges(
        "evaluator",
        _route_after_evaluator,
        {"finalize": "comparador", "retry": "prepare_retry"},
    )
    graph.add_edge("prepare_retry", "rag")
    graph.add_edge("comparador", "coordinator_finalize")
    graph.add_edge("coordinator_finalize", END)

    return graph.compile(checkpointer=checkpointer)


def run_agentic(  # noqa: PLR0913 -- parametros nombrados, entrada principal de la API
    question: str,
    cfg: DictConfig,
    pipeline: QueryPipeline,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    thread_id: str | None = None,
    resume_answer: str | None = None,
) -> RagAnswer | PendingTreeQuestion:
    """Punto de entrada usado por la API (ver api/main.py, flag GPC_USE_AGENTS).

    - Conversacion nueva: se llama con `question` (y sin `resume_answer`).
    - Continuar un wizard pendiente: se llama con `thread_id` y
      `resume_answer` (la respuesta del usuario a la ultima pregunta del
      agente arbol); `question` se ignora en ese caso, se recupera del
      estado persistido en el checkpointer.
    - `checkpointer` debe ser el MISMO objeto (p.ej. un `MemorySaver` a
      nivel de modulo en api/main.py) entre la llamada que pausa y la que
      reanuda -- si no, el estado pausado no se encuentra. Si no se pasa
      ninguno (p.ej. en tests, o fuera de la API), se crea uno nuevo por
      llamada: sirve igual mientras el grafo no quede pausado entre
      llamadas distintas.
    """
    checkpointer = checkpointer or MemorySaver()
    app = build_agent_graph(cfg, pipeline, checkpointer=checkpointer)

    if resume_answer is not None:
        if not thread_id:
            raise ValueError("resume_answer requiere thread_id")
        graph_config = {"configurable": {"thread_id": thread_id}}
        app.invoke(Command(resume=resume_answer), config=graph_config)
    else:
        thread_id = thread_id or str(uuid.uuid4())
        graph_config = {"configurable": {"thread_id": thread_id}}
        app.invoke({"question": question, "retry_count": 0}, config=graph_config)

    snapshot = app.get_state(graph_config)

    # IMPORTANTE: `snapshot.next` NO es confiable para detectar una pausa
    # despues de ya haber reanudado al menos una vez (queda vacio aunque
    # todavia haya un interrupt() pendiente) -- verificado empiricamente
    # (ver sesion de depuracion). La señal correcta es si alguna tarea en
    # `snapshot.tasks` tiene interrupciones pendientes.
    pending_interrupts = [i for task in snapshot.tasks for i in task.interrupts]

    if pending_interrupts:
        # El grafo sigue pausado: el agente arbol tiene otra pregunta.
        pending = pending_interrupts[0].value
        logger.info("Agente arbol: pregunta pendiente (thread_id=%s)", thread_id)
        return PendingTreeQuestion(thread_id=thread_id, question=pending["question"])

    result_state = snapshot.values
    return RagAnswer(
        question=result_state.get("question", question),
        answer=result_state["final_answer"],
        sources=result_state.get("final_sources", []),
    )
