"""Integrador maestro: compone UNA sola respuesta a partir de las aportaciones
que el verificador declaro elegibles (ver agents/verifier.py).

Reemplaza al antiguo comparador, que elegia una sola aportacion (el arbol si
tenia match, si no el RAG). Ahora:

- Si ninguna aportacion es elegible: se informa la abstencion del RAG (si la
  hubo) o el mensaje de respaldo, sin fuentes.
- Si solo una es elegible: se publica tal cual, sin pasar por el LLM.
- Si ambas son elegibles (arbol + RAG): un modelo de lenguaje local redacta
  una respuesta consolidada usando EXCLUSIVAMENTE esas aportaciones, y una
  segunda llamada verifica la fidelidad de la sintesis contra ellas. Si la
  sintesis falla, no es interpretable o no supera la verificacion (falla
  cerrada), se publica la composicion determinista: las aportaciones
  verificadas en bloques, sin redaccion nueva.

En todos los casos se informan las discrepancias detectadas por el
verificador, y nunca se agregan hechos clinicos nuevos. El estado de cada
agente se informa en su propio panel (build_agent_panels), que el chat
muestra junto a la sugerencia consolidada.

El integrador NO le habla al usuario directamente: deja su resultado en
state["integrated_*"] y el coordinador (coordinator.build_final_output)
sigue siendo el unico punto de salida.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ollama import Client
from omegaconf import DictConfig

from gpc_rag.agents.schemas import FIDELITY_SCHEMA
from gpc_rag.agents.state import AgentState
from gpc_rag.agents.timeouts import llm_timeout
from gpc_rag.common.models import AgentPanel, Chunk, RetrievedChunk
from gpc_rag.common.refusal import is_refusal

logger = logging.getLogger(__name__)

FALLBACK_NO_ELIGIBLE = (
    "No puedo confirmar esta respuesta con suficiente seguridad a partir de "
    "las guias disponibles. Te recomiendo consultar directamente la guia "
    "original o a un profesional clinico."
)

_FIDELITY_ATTEMPTS = 2

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

_SYNTHESIS_SYSTEM_PROMPT = """\
Eres el integrador de un asistente clinico institucional. Recibes la \
pregunta del usuario y dos APORTACIONES ya verificadas:
1. ARBOL: conclusion de una regla auditada del protocolo institucional.
2. RAG: respuesta basada en fragmentos recuperados de los protocolos.
El ARBOL proviene del arbol clinico aprobado por el comite de la clinica.

Redacta UNA sola respuesta en español, clara y breve, que integre ambas \
aportaciones. Reglas obligatorias:
- Usa EXCLUSIVAMENTE informacion que aparezca en las aportaciones. No \
agregues farmacos, dosis, cifras, criterios ni recomendaciones nuevas.
- Conserva las referencias: menciona el protocolo del ARBOL, y \
mantiene las citas de archivo, seccion y pagina que traiga el RAG.
- La conclusion del ARBOL es la recomendacion institucional auditada: \
presentala primero. Usa el RAG para complementar o contextualizar.
- Si ambas aportaciones dicen cosas distintas, o si hay NOTAS DEL \
VERIFICADOR, no elijas una ni inventes una conciliacion: menciona la \
diferencia y de donde viene cada una.
- No uses conocimiento propio ni cites fuentes externas.

Responde solo con el texto de la respuesta integrada.
"""

_FIDELITY_SYSTEM_PROMPT = """\
Eres un auditor de fidelidad. Compara una RESPUESTA INTEGRADA con las \
APORTACIONES de las que debia salir. No respondas la pregunta.

Rechaza la respuesta integrada si:
- Agrega un farmaco, dosis, cifra, criterio, recomendacion o fuente que NO \
aparece en las APORTACIONES.
- Omite o contradice la conclusion del ARBOL.
- Cambia el significado de alguna aportacion.

Apruebala si solo reorganiza o resume lo que dicen las APORTACIONES.

Responde UNICAMENTE con un objeto JSON, sin texto adicional:
- Si se aprueba: {"verdict": "OK"}
- Si se rechaza: {"verdict": "RECHAZADA", "reason": "<motivo breve en español>"}
"""


def tree_result_as_sources(tree_result: dict) -> list[RetrievedChunk]:
    """Convierte las citas del arbol al formato RetrievedChunk que ya usa el
    chat para mostrar fuentes, sin cambiar el esquema de la API."""
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


def _deterministic_composition(tree: dict, rag: dict) -> str:
    return (
        "**Según el protocolo institucional (árbol de decisión):**\n"
        f"{tree['content'].strip()}\n\n"
        "**Según los documentos recuperados (RAG):**\n"
        f"{rag['content'].strip()}"
    )


def _contributions_prompt(
    question: str, tree: dict, rag: dict, discrepancies: list[str] | None = None
) -> str:
    prompt = (
        f"PREGUNTA:\n{question}\n\n"
        f"APORTACION ARBOL:\n{tree['content']}\n\n"
        f"APORTACION RAG:\n{rag['content']}"
    )
    if discrepancies:
        prompt += "\n\nNOTAS DEL VERIFICADOR:\n" + "\n".join(
            f"- {d}" for d in discrepancies
        )
    return prompt


def _model(cfg: DictConfig) -> str:
    return cfg.agents.get("integrator_model") or cfg.agents.coordinator_model


def _synthesize(
    question: str, tree: dict, rag: dict, discrepancies: list[str], cfg: DictConfig
) -> str | None:
    client = Client(host=cfg.env.ollama_url, timeout=llm_timeout(cfg))
    response = client.chat(
        model=_model(cfg),
        messages=[
            {"role": "system", "content": _SYNTHESIS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _contributions_prompt(question, tree, rag, discrepancies),
            },
        ],
        options={"temperature": 0.1, "num_predict": 600},
        stream=False,
    )
    text = (response["message"]["content"] or "").strip()
    return text or None


def _check_fidelity(
    question: str, tree: dict, rag: dict, synthesis: str, cfg: DictConfig
) -> bool:
    """Falla cerrada: solo un veredicto OK explicito aprueba la sintesis."""
    client = Client(host=cfg.env.ollama_url, timeout=llm_timeout(cfg))
    user_prompt = (
        f"{_contributions_prompt(question, tree, rag)}\n\n"
        f"RESPUESTA INTEGRADA A AUDITAR:\n{synthesis}"
    )
    for attempt in range(1, _FIDELITY_ATTEMPTS + 1):
        response = client.chat(
            model=_model(cfg),
            messages=[
                {"role": "system", "content": _FIDELITY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            format=FIDELITY_SCHEMA,
            options={"temperature": 0.0, "num_predict": 300},
            stream=False,
        )
        content = response["message"]["content"].strip()
        try:
            verdict = str(json.loads(content).get("verdict", "")).strip().upper()
        except (json.JSONDecodeError, AttributeError):
            logger.warning(
                "Integrador: veredicto de fidelidad no interpretable (intento %d/%d): %r",
                attempt,
                _FIDELITY_ATTEMPTS,
                content,
            )
            continue
        if verdict == "OK":
            return True
        if verdict == "RECHAZADA":
            logger.info(
                "Integrador: la sintesis no supero la verificacion de fidelidad: %r",
                content,
            )
            return False
    return False


def integrate(state: AgentState, cfg: DictConfig) -> dict:
    contributions = state.get("contributions") or []
    tree = next(c for c in contributions if c["agent"] == "arbol")
    rag = next(c for c in contributions if c["agent"] == "rag")
    discrepancies = state.get("discrepancies") or []

    tree_sources = (
        tree_result_as_sources(state["tree_result"]) if tree["eligible"] else []
    )
    rag_sources = list(state.get("sources", []) or []) if rag["eligible"] else []

    if tree["eligible"] and rag["eligible"]:
        mode = "composicion"
        body = _deterministic_composition(tree, rag)
        try:
            synthesis = _synthesize(state["question"], tree, rag, discrepancies, cfg)
            if (
                synthesis
                and not is_refusal(synthesis)
                and _check_fidelity(state["question"], tree, rag, synthesis, cfg)
            ):
                mode, body = "sintesis", synthesis
        except Exception:  # un fallo del modelo no debe tumbar la respuesta
            logger.exception(
                "Integrador: fallo la sintesis; se usa la composicion determinista."
            )
        logger.info("Integrador: arbol + RAG, modalidad=%s", mode)
        return {
            "integration_mode": mode,
            "integrated_answer": _with_annotations(body, discrepancies),
            "integrated_sources": tree_sources + rag_sources,
        }

    if tree["eligible"] or rag["eligible"]:
        single = tree if tree["eligible"] else rag
        logger.info("Integrador: una sola aportacion elegible (%s)", single["agent"])
        return {
            "integration_mode": "unica",
            "integrated_answer": _with_annotations(single["content"], discrepancies),
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
