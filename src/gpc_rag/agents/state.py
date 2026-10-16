"""Estado compartido entre los agentes del grafo.

Ver agents/graph.py para el flujo completo. Se usa TypedDict (no dataclass)
porque es lo que LangGraph espera como esquema de estado de un StateGraph.
"""

from __future__ import annotations

from typing import Any, TypedDict

from gpc_rag.common.models import RetrievedChunk


class AgentState(TypedDict, total=False):
    question: str

    # Agente coordinador (pre-filtro de admision)
    is_appropriate: bool
    rejection_reason: str | None

    # Agente arbol (tree_prepare_node + tree_agent_node, ver tree_agent.py):
    # evalua deterministicamente contra cpg_tree, con preguntas tipo wizard
    # via interrupt() cuando faltan datos criticos.
    tree_protocol_id: str | None  # "NAC" | "ITU" | None (ningun protocolo aplica)
    tree_mode: str  # "caso" (evalua datos del paciente) | "consulta" (reglas relacionadas)
    tree_initial_values: dict[str, Any]  # primera extraccion, antes de preguntar
    tree_result: dict[str, Any] | None  # ver tree_agent._render_tree_result
    tree_questions_asked: int
    tree_outcome: str | None  # "informacion_faltante" | "ausencia_de_coincidencias" sin match
    tree_error: str | None  # error tecnico de la rama del arbol

    # Agente RAG (retrieval + generacion)
    answer: str
    sources: list[RetrievedChunk]
    extra_instruction: str | None  # se llena si el evaluador pide reintentar
    rag_error: str | None  # error tecnico o tiempo maximo excedido en la rama del RAG

    # Agente evaluador (guardrail en linea, antes de responder al usuario)
    eval_ok: bool
    eval_status: str  # "aprobada" | "rechazada" | "no_verificable" (falla cerrada)
    eval_feedback: str | None
    retry_count: int

    # Agente verificador de evidencia (ver agents/verifier.py): una
    # aportacion por agente con un contrato comun (agent, status, eligible,
    # content, references, detail) y las discrepancias de version detectadas.
    contributions: list[dict[str, Any]]
    discrepancies: list[str]

    # Integrador maestro (ver agents/integrator.py): respuesta consolidada.
    # El coordinador sigue siendo el UNICO que arma lo que ve el usuario
    # (build_final_output), a partir de lo que el integrador deja aqui.
    integration_mode: str  # "sintesis" | "composicion" | "unica" | "sin_aportaciones"
    integrated_answer: str
    integrated_sources: list[RetrievedChunk]

    # Salida final: la arma el coordinador (unico agente que "habla" con el
    # usuario), decide si se muestran fuentes o no segun el resultado.
    final_answer: str
    final_sources: list[RetrievedChunk]
