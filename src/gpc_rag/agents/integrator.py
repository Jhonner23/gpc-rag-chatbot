"""Integrador: reune las aportaciones que el verificador declaro elegibles
(ver agents/verifier.py) sin redactar texto nuevo.

El chat muestra la aportacion de cada agente en su propio panel
(build_agent_panels) y, debajo, las notas del verificador (contradicciones o
versiones distintas entre fuentes). No hay respuesta consolidada redactada
por un modelo: en pruebas reales con modelos locales de 7B, la sintesis
mezclaba condiciones entre ramas y la verificacion de fidelidad no lo
detectaba, asi que se retiro.

`integrated_answer` (el campo `answer` de la API) es la composicion literal
de las aportaciones verificadas, en bloques:
- Ninguna elegible: la abstencion del RAG (si la hubo) o el mensaje de
  respaldo, sin fuentes.
- Una o dos elegibles: su texto tal cual, cada una con su encabezado.

El integrador NO le habla al usuario directamente: deja su resultado en
state["integrated_*"] y el coordinador (coordinator.build_final_output)
sigue siendo el unico punto de salida.
"""

from __future__ import annotations

import logging
from typing import Any

from omegaconf import DictConfig

from gpc_rag.agents.state import AgentState
from gpc_rag.common.models import AgentPanel, Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

FALLBACK_NO_ELIGIBLE = (
    "No puedo confirmar esta respuesta con suficiente seguridad a partir de "
    "las guias disponibles. Te recomiendo consultar directamente la guia "
    "original o a un profesional clinico."
)

# Etiquetas del estado de cada agente y de la verificacion. No deben
# contener los marcadores de common/refusal.py (p. ej. "no encontr",
# "informacion suficiente"): el chat usa esos marcadores para ocultar las
# fuentes.
_STATUS_LABELS = {
    "respuesta_fundamentada": "respuesta fundamentada",
    "abstencion": "abstención",
    "informacion_faltante": "información faltante",
    "ausencia_de_coincidencias": "ausencia de coincidencias",
    "error_tecnico": "error técnico",
}
_VERIFICATION_LABELS = {
    "verificada": "verificada",
    "sin_procedencia": "no publicada: sin procedencia verificable",
    "rechazada_por_evaluador": "no publicada: rechazada por el evaluador",
    "no_verificable": "no publicada: no fue posible verificarla",
    "no_relacionada": "no publicada: no relacionada con la pregunta",
}

def tree_result_as_sources(tree_result: dict) -> list[RetrievedChunk]:
    """Convierte las citas del arbol al formato RetrievedChunk que ya usa el
    chat para mostrar fuentes, sin cambiar el esquema de la API."""
    sources = []
    for citation in tree_result.get("citations", []):
        chunk = Chunk(
            text=citation.get("verbatim_text") or "",
            source_file=citation["source_file"],
            section=citation.get("section") or tree_result["rule_id"],
            # El diagrama es de una sola pagina: la referencia util para el
            # usuario es la pagina del protocolo, que ya va en la seccion.
            page=None,
        )
        sources.append(RetrievedChunk(chunk=chunk, score=1.0, retrieval_method="arbol"))
    return sources


def _agent_label(contribution: dict[str, Any]) -> str:
    label = _STATUS_LABELS.get(contribution["status"], contribution["status"])
    verification = contribution.get("verification")
    if verification:
        label += f" ({_VERIFICATION_LABELS.get(verification, verification)})"
    return label


def _with_annotations(body: str, discrepancies: list[str]) -> str:
    parts = [body.strip()]
    if discrepancies:
        parts.append("**Nota sobre las fuentes:** " + " ".join(discrepancies))
    return "\n\n".join(parts)


_BLOCK_TITLES = {
    "arbol": "Según el árbol clínico del protocolo institucional:",
    "rag": "Según los documentos recuperados (RAG):",
}


def _composition(eligible: list[dict[str, Any]]) -> str:
    if len(eligible) == 1:
        return eligible[0]["content"].strip()
    return "\n\n".join(f"**{_BLOCK_TITLES[c['agent']]}**\n{c['content'].strip()}" for c in eligible)


def integrate(state: AgentState, cfg: DictConfig) -> dict:
    contributions = state.get("contributions") or []
    tree = next(c for c in contributions if c["agent"] == "arbol")
    rag = next(c for c in contributions if c["agent"] == "rag")
    discrepancies = state.get("discrepancies") or []

    tree_sources = tree_result_as_sources(state["tree_result"]) if tree["eligible"] else []
    rag_sources = list(state.get("sources", []) or []) if rag["eligible"] else []
    eligible = [c for c in (tree, rag) if c["eligible"]]

    if eligible:
        mode = "composicion" if len(eligible) > 1 else "unica"
        logger.info("Integrador: aportaciones elegibles=%s", [c["agent"] for c in eligible])
        return {
            "integration_mode": mode,
            "integrated_answer": _with_annotations(_composition(eligible), discrepancies),
            "integrated_sources": tree_sources + rag_sources,
        }

    body = rag["content"] if rag["status"] == "abstencion" else FALLBACK_NO_ELIGIBLE
    logger.info(
        "Integrador: ninguna aportacion elegible (arbol=%s rag=%s)",
        tree["status"],
        rag["status"],
    )
    return {
        "integration_mode": "sin_aportaciones",
        "integrated_answer": _with_annotations(body, []),
        "integrated_sources": [],
    }


# Paneles por agente ---------------------------------------------------------

_PANEL_TITLES = {
    "arbol": "Árbol de decisión",
    "rag": "RAG (documentos de los protocolos)",
}

# Texto del panel cuando el agente no tiene una aportacion publicable. Las
# aportaciones no verificadas NO se muestran (falla cerrada): solo se
# explica por que el agente no llego a una respuesta.
_TREE_NO_ANSWER = {
    "abstencion": "ningún protocolo institucional aplica a la pregunta.",
    "informacion_faltante": "faltaron datos del caso para recorrer el árbol.",
    "ausencia_de_coincidencias": "ninguna rama del árbol aplica al caso.",
    "error_tecnico": "ocurrió un error técnico.",
}
_RAG_NO_ANSWER = {
    "ausencia_de_coincidencias": "No se recuperaron fragmentos relacionados con la pregunta.",
    "error_tecnico": "El RAG no pudo completarse por un error técnico.",
}
_HIDDEN_REASON = {
    "sin_procedencia": "no tiene procedencia verificable",
    "rechazada_por_evaluador": "fue rechazada por el evaluador",
    "no_verificable": "no fue posible verificarla",
    "no_relacionada": "no está relacionada con la pregunta",
}


def _tree_panel_text(tree: dict[str, Any]) -> str:
    if tree["eligible"]:
        return tree["content"]
    if tree["status"] == "respuesta_fundamentada":
        reason = _HIDDEN_REASON.get(tree.get("verification"), "no fue verificada")
        return f"El árbol de decisión no llegó a una respuesta: su aportación {reason}."
    if tree["status"] in {"informacion_faltante", "ausencia_de_coincidencias"} and tree.get("detail"):
        return f"El árbol de decisión no llegó a una respuesta: {tree['detail']}"
    reason = _TREE_NO_ANSWER.get(tree["status"], tree["status"])
    return f"El árbol de decisión no llegó a una respuesta: {reason}"


def _rag_panel_text(rag: dict[str, Any]) -> str:
    if rag["eligible"]:
        return rag["content"]
    if rag["status"] == "abstencion" and rag.get("content"):
        return rag["content"]
    if rag["status"] == "respuesta_fundamentada":
        reason = _HIDDEN_REASON.get(rag.get("verification"), "no fue verificada")
        return f"La respuesta del RAG no se muestra porque {reason}."
    return _RAG_NO_ANSWER.get(rag["status"], rag["status"])


def build_agent_panels(state: AgentState) -> list[AgentPanel]:
    """Un panel por agente, a partir de las aportaciones del verificador.

    No llama a ningun modelo: solo presenta lo que ya decidio el verificador.
    Vacio si no hubo aportaciones (p. ej. el coordinador rechazo la
    pregunta). Cuando exista el agente de casos similares, su aportacion
    tendra su propio panel con el mismo contrato.
    """
    contributions = state.get("contributions") or []
    by_agent = {c["agent"]: c for c in contributions}
    panels: list[AgentPanel] = []

    tree = by_agent.get("arbol")
    if tree is not None:
        panels.append(
            AgentPanel(
                agent="arbol",
                title=_PANEL_TITLES["arbol"],
                status=_agent_label(tree),
                eligible=tree["eligible"],
                content=_tree_panel_text(tree),
                sources=(
                    tree_result_as_sources(state["tree_result"]) if tree["eligible"] else []
                ),
            )
        )

    rag = by_agent.get("rag")
    if rag is not None:
        panels.append(
            AgentPanel(
                agent="rag",
                title=_PANEL_TITLES["rag"],
                status=_agent_label(rag),
                eligible=rag["eligible"],
                content=_rag_panel_text(rag),
                sources=list(state.get("sources", []) or []) if rag["eligible"] else [],
            )
        )
    return panels
