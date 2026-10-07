"""Agente comparador: decide si la respuesta final sale del arbol o del RAG.

Regla de decision (pedida explicitamente: "uno que compare la respuesta del
arbol y el RAG y escoja la mejor"):

- Si el agente arbol encontro una regla MATCHED con validation_status
  EXTRACTED (ver tree_agent.py / tree_support.py -- nunca UNRESOLVED), se
  prefiere esa respuesta: es deterministica, auditada, y cita exactamente la
  regla y el fragmento fuente del protocolo institucional.
- Si no, se usa la respuesta del RAG (ya pasada por el evaluador).

El comparador NO le habla al usuario directamente -- deja su decision en
state["chosen_*"] y es el coordinador (coordinator.build_final_output) quien
arma la respuesta final, siguiendo la misma regla de "un solo punto de
salida" que ya aplica el grafo.
"""

from __future__ import annotations

import logging

from gpc_rag.agents.state import AgentState
from gpc_rag.common.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


def _tree_result_as_sources(tree_result: dict) -> list[RetrievedChunk]:
    """Convierte las citas del arbol al mismo formato RetrievedChunk que usa
    el RAG, para que el chat las muestre con el `_format_sources` que ya
    existe, sin tener que tocar el esquema de la API."""
    sources = []
    for citation in tree_result.get("citations", []):
        chunk = Chunk(
            text=citation.get("verbatim_text") or "",
            source_file=citation["source_file"],
            section=citation.get("section") or tree_result["rule_id"],
            page=citation.get("page"),
        )
        sources.append(RetrievedChunk(chunk=chunk, score=1.0, retrieval_method="arbol"))
    return sources


def compare_and_choose(state: AgentState) -> dict:
    tree_result = state.get("tree_result")

    if tree_result:
        logger.info(
            "Comparador: se usa el arbol (protocolo=%s regla=%s)",
            tree_result.get("protocol_id"),
            tree_result.get("rule_id"),
        )
        return {
            "chosen_source": "arbol",
            "chosen_answer": tree_result["answer_text"],
            "chosen_sources": _tree_result_as_sources(tree_result),
        }

    logger.info("Comparador: el arbol no llego a una conclusion auditada; se usa el RAG.")
    return {
        "chosen_source": "rag",
        "chosen_answer": state.get("answer", ""),
        "chosen_sources": state.get("sources", []),
    }
