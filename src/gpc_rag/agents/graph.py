"""Grafo de agentes (LangGraph).

    START
      |
      v
coordinator_filter --(no adecuada)----------------------------------+
      |                                                              |
   (adecuada, fan-out en paralelo)                                   |
      |                                                              |
      +-------------------+                                          |
      |                   |                                          |
      v                   v                                          |
  tree_prepare           rag  <----------------------+                |
      |                   |                           | (reintento,   |
      v                   v                           |  max 1 vez,   |
  tree_agent          evaluator ------------------------+  solo si fue  |
  (interrupt() ->         |                              rechazada)    |
   wizard si faltan   (aprobada, no verificable, o sin reintentos)      |
   datos)                 |                                             |
      |                   v                                            |
      |               rag_done  (nodo de sincronizacion, sin efecto)    |
      |                   |                                            |
      +---------+---------+                                            |
                v                                                       |
           verificador  (estado, procedencia y discrepancias de cada    |
                         aportacion; contradicciones con LLM)           |
                v                                                       |
           integrador  (respuesta consolidada con LLM + verificacion    |
                        de fidelidad; si falla, composicion determinista)|
                v                                                       v
                      coordinator_finalize --> END

El coordinador es el UNICO agente que decide que ve el usuario final
(agents/coordinator.py: build_final_output) -- ni el agente RAG, el
evaluador, el agente arbol, el verificador ni el integrador devuelven nada
directamente.

El agente arbol puede pausar el grafo a mitad de camino (LangGraph
`interrupt()`) para preguntarle al usuario, una variable a la vez, estilo
wizard -- ver agents/tree_agent.py. Esa pausa/resume se identifica con un
`thread_id` que persiste en el checkpointer (ver run_agentic): la API
(api/main.py) se lo devuelve al chat, y el chat se lo reenvia junto con la
respuesta a la pregunta pendiente.

Nota sobre el paralelismo (tree_agent / rag): ambas ramas se lanzan desde
`coordinator_filter` en el mismo "super-step" de LangGraph. Si `tree_agent`
pausa con `interrupt()`, el grafo completo se detiene al terminar ese
super-step -- cualquier super-step posterior de la rama RAG (p.ej. un
reintento) NO se ejecuta hasta que el usuario responda TODAS las preguntas
pendientes del arbol y esa rama quede resuelta sin mas interrupciones. El
nodo `rag_done` es el punto de union: `verificador` solo se dispara cuando
`tree_agent` Y `rag_done` ya completaron.
"""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from omegaconf import DictConfig

from gpc_rag.agents.coordinator import build_final_output, filter_query
from gpc_rag.agents.evaluator import NO_VERIFICABLE, evaluate_answer_status
from gpc_rag.agents.integrator import build_agent_panels, integrate
from gpc_rag.agents.state import AgentState
from gpc_rag.agents.timeouts import rag_timeout
from gpc_rag.agents.tree_agent import tree_agent_node, tree_prepare_node
from gpc_rag.agents.verifier import verify_contributions
from gpc_rag.common.models import PendingTreeQuestion, RagAnswer
from gpc_rag.pipelines.query_pipeline import QueryPipeline

logger = logging.getLogger(__name__)

MAX_RETRIES = 1


def _coordinator_filter_node(state: AgentState, cfg: DictConfig) -> dict:
    try:
        is_ok, reason = filter_query(state["question"], cfg)
    except Exception:
        # Igual que ante un veredicto ilegible: admitir la pregunta no publica
        # contenido clinico, todavia debe pasar por la verificacion.
        logger.exception("Coordinador (filtro): error tecnico; se admite la pregunta.")
        is_ok, reason = True, None
    logger.info("Coordinador (filtro): pregunta=%r adecuada=%s", state["question"][:60], is_ok)
    return {"is_appropriate": is_ok, "rejection_reason": reason}


def _rag_node(state: AgentState, pipeline: QueryPipeline, cfg: DictConfig) -> dict:
    """Rama RAG con tiempo maximo. Si se supera, la rama queda en error
    tecnico y el resultado tardio se descarta (el hilo puede terminar en
    segundo plano, pero su resultado nunca se asocia a esta ni a otra
    consulta)."""
    timeout = rag_timeout(cfg)
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(
        pipeline.answer, state["question"], extra_instruction=state.get("extra_instruction")
    )
    try:
        result = future.result(timeout=timeout)
    except FutureTimeoutError:
        logger.error("Agente RAG: supero el tiempo maximo de %.0f s.", timeout)
        return {"answer": "", "sources": [], "rag_error": f"Tiempo maximo de {timeout:.0f} s excedido."}
    except Exception as exc:
        logger.exception("Agente RAG: error tecnico.")
        return {"answer": "", "sources": [], "rag_error": f"Error tecnico: {type(exc).__name__}."}
    finally:
        executor.shutdown(wait=False)
    return {"answer": result.answer, "sources": result.sources, "rag_error": None}


def _tree_prepare_node(state: AgentState, cfg: DictConfig) -> dict:
    try:
        return tree_prepare_node(state, cfg)
    except Exception as exc:
        logger.exception("Agente arbol (preparacion): error tecnico.")
        return {
            "tree_protocol_id": None,
            "tree_initial_values": {},
            "tree_error": f"Error tecnico: {type(exc).__name__}.",
        }


def _tree_agent_node(state: AgentState, cfg: DictConfig) -> dict:
    if state.get("tree_error"):
        return {"tree_result": None, "tree_questions_asked": 0, "tree_stop_detail": None}
    try:
        return tree_agent_node(state, cfg)
    except GraphBubbleUp:
        raise  # interrupt() del wizard: no es un error, LangGraph lo maneja
    except Exception as exc:
        logger.exception("Agente arbol: error tecnico.")
        return {
            "tree_result": None,
            "tree_questions_asked": 0,
            "tree_error": f"Error tecnico: {type(exc).__name__}.",
        }


def _evaluator_node(state: AgentState, cfg: DictConfig) -> dict:
    try:
        status, feedback = evaluate_answer_status(
            state["question"], state.get("sources", []), state["answer"], cfg
        )
    except Exception as exc:
        logger.exception("Evaluador: error tecnico; la respuesta queda como no verificable.")
        status, feedback = NO_VERIFICABLE, f"Error tecnico del evaluador: {type(exc).__name__}."
    logger.info(
        "Evaluador: estado=%s feedback=%r retry_count=%d",
        status,
        feedback,
        state.get("retry_count", 0),
    )
    return {"eval_ok": status == "aprobada", "eval_status": status, "eval_feedback": feedback}


def _prepare_retry_node(state: AgentState) -> dict:
    feedback = state.get("eval_feedback") or "la respuesta no estaba bien fundamentada."
    instruction = (
        f"Tu respuesta anterior fue rechazada por este motivo: {feedback} "
        "Vuelve a responder siendo mas literal: usa solo texto que aparezca "
        "en el CONTEXTO, y si no hay informacion suficiente, dilo "
        "explicitamente en vez de completar con conocimiento propio."
    )
    return {"extra_instruction": instruction, "retry_count": state.get("retry_count", 0) + 1}


def _rag_done_node(state: AgentState) -> dict:  # noqa: ARG001 -- nodo de sincronizacion
    """Nodo sin efecto sobre el estado: solo sirve de punto de union para
    que `verificador` espere a que la rama RAG (incluido su loop de
    reintento) haya terminado, en paralelo con la rama del arbol."""
    return {}


def _coordinator_finalize_node(state: AgentState) -> dict:
    final_answer, final_sources = build_final_output(state)
    return {"final_answer": final_answer, "final_sources": final_sources}


def _route_after_filter(state: AgentState) -> list[str]:
    if state.get("is_appropriate"):
        return ["tree_prepare", "rag"]
    return ["coordinator_finalize"]


def _route_after_rag(state: AgentState) -> str:
    # Si la rama fallo no hay respuesta que auditar.
    return "skip" if state.get("rag_error") else "evaluate"


def _route_after_evaluator(state: AgentState) -> str:
    if state.get("eval_ok"):
        return "finalize"
    # Un veredicto ilegible es un problema del juez, no de la respuesta:
    # regenerar el RAG no lo arregla, asi que no se reintenta.
    if state.get("eval_status") == NO_VERIFICABLE:
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
    graph.add_node("tree_prepare", lambda s: _tree_prepare_node(s, cfg))
    graph.add_node("tree_agent", lambda s: _tree_agent_node(s, cfg))
    graph.add_node("rag", lambda s: _rag_node(s, pipeline, cfg))
    graph.add_node("evaluator", lambda s: _evaluator_node(s, cfg))
    graph.add_node("prepare_retry", _prepare_retry_node)
    graph.add_node("rag_done", _rag_done_node)
    graph.add_node("verificador", lambda s: verify_contributions(s, cfg))
    graph.add_node("integrador", lambda s: integrate(s, cfg))
    graph.add_node("coordinator_finalize", _coordinator_finalize_node)

    graph.set_entry_point("coordinator_filter")
    graph.add_conditional_edges("coordinator_filter", _route_after_filter)
    graph.add_edge("tree_prepare", "tree_agent")
    graph.add_conditional_edges(
        "rag", _route_after_rag, {"evaluate": "evaluator", "skip": "rag_done"}
    )
    graph.add_conditional_edges(
        "evaluator",
        _route_after_evaluator,
        {"finalize": "rag_done", "retry": "prepare_retry"},
    )
    graph.add_edge("prepare_retry", "rag")

    # Punto de sincronizacion: el verificador no corre hasta que la rama del
    # arbol (tree_agent) Y la rama RAG (rag_done) hayan terminado.
    graph.add_edge(["tree_agent", "rag_done"], "verificador")
    graph.add_edge("verificador", "integrador")
    graph.add_edge("integrador", "coordinator_finalize")
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
        panels=build_agent_panels(result_state),
        notes=list(result_state.get("discrepancies") or []),
    )
