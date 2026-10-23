"""Agente verificador de evidencia: decide que aportaciones son elegibles para
la respuesta final, ANTES de integrarlas.

Corre una sola vez, cuando ya terminaron la rama del arbol (tree_agent) y la
rama del RAG (rag -> evaluator -> rag_done).

Por cada agente produce una "aportacion" con un contrato comun:

    {
        "agent": "arbol" | "rag",
        "status": <estado del agente, ver abajo>,
        "verification": <resultado de la verificacion, o None>,
        "eligible": bool,      # si puede entrar a la respuesta integrada
        "content": str,        # texto de la aportacion (vacio si no hay)
        "references": [...],   # procedencia especifica de cada agente
        "detail": str | None,  # motivo legible del estado o de la verificacion
    }

Estados de cada agente (los mismos para todos):
- respuesta_fundamentada:     el agente produjo una respuesta con su evidencia.
- abstencion:                 el agente declina responder (ningun protocolo
                              aplica, o el RAG reconoce que las fuentes no
                              cubren el tema).
- informacion_faltante:       el arbol agoto las preguntas sin datos suficientes.
- ausencia_de_coincidencias:  ninguna rama del arbol aplica al caso, o el RAG no
                              recupero fragmentos.
- error_tecnico:              la rama fallo o supero su tiempo maximo.

Resultado de la verificacion (solo para respuesta_fundamentada):
- verificada:              procedencia completa (y, en el RAG, aprobada por el
                           evaluador).
- sin_procedencia:         faltan archivo, pagina o texto de la fuente.
- rechazada_por_evaluador: el RAG no supero la auditoria tras el reintento.
- no_verificable:          el evaluador no produjo un veredicto interpretable
                           (falla cerrada: no se publica).
- no_relacionada:          la aportacion del arbol no responde a la pregunta
                           (juzgado por un modelo local; falla cerrada).

Discrepancias (se conservan con su procedencia, NO se resuelven):
- de version: el arbol y el RAG citan el mismo protocolo institucional (p. ej.
  CT-PL-193) con versiones distintas en el nombre del archivo (determinista).
- de contenido: si ambas aportaciones son elegibles, un modelo local juzga si
  se contradicen (acciones incompatibles o hechos opuestos). Si su veredicto
  no es interpretable, se informa que no fue posible comprobarlo.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ollama import Client
from omegaconf import DictConfig

from gpc_rag.agents.schemas import CONTRADICTION_SCHEMA, RELEVANCE_SCHEMA
from gpc_rag.agents.state import AgentState
from gpc_rag.agents.timeouts import llm_timeout
from gpc_rag.common.llm_options import ollama_options
from gpc_rag.common.models import RetrievedChunk
from gpc_rag.common.refusal import is_refusal

logger = logging.getLogger(__name__)

_LOG_SNIPPET_CHARS = 2000

_PROTOCOL_CODE_RE = re.compile(r"CT-PL-\d+", re.IGNORECASE)
_VERSION_RE = re.compile(r"v(\d+)", re.IGNORECASE)
_CONTRADICTION_ATTEMPTS = 2

_CONTRADICTION_SYSTEM_PROMPT = """\
Eres un auditor que compara dos aportaciones sobre la misma pregunta clinica:
- ARBOL: recorrido o nodos del arbol clinico del protocolo institucional.
- RAG: respuesta basada en fragmentos recuperados de los protocolos.

Determina si se CONTRADICEN: recomiendan acciones incompatibles entre si, o \
afirman hechos opuestos. Que una sea mas detallada, que traten aspectos \
distintos o que se complementen NO es una contradiccion. No respondas la \
pregunta ni uses conocimiento propio.

Responde UNICAMENTE con un objeto JSON, sin texto adicional:
- Si no se contradicen: {"contradiccion": false}
- Si se contradicen: {"contradiccion": true, "detalle": "<en que se \
contradicen, en una frase, en español>"}
"""


def _protocol_versions(filenames: list[str]) -> dict[str, set[str]]:
    """{codigo_protocolo: {versiones}} a partir de nombres de archivo."""
    found: dict[str, set[str]] = {}
    for name in filenames:
        code = _PROTOCOL_CODE_RE.search(name or "")
        version = _VERSION_RE.search(name or "")
        if code and version:
            found.setdefault(code.group(0).upper(), set()).add(
                str(int(version.group(1)))
            )
    return found


def _contribution(agent: str, status: str, **kwargs: Any) -> dict[str, Any]:
    base = {
        "agent": agent,
        "status": status,
        "verification": None,
        "eligible": False,
        "content": "",
        "references": [],
        "detail": None,
    }
    base.update(kwargs)
    return base


def _verify_tree(state: AgentState) -> dict[str, Any]:
    if state.get("tree_error"):
        return _contribution("arbol", "error_tecnico", detail=state.get("tree_error"))
    if not state.get("tree_protocol_id"):
        return _contribution(
            "arbol", "abstencion", detail="Ningun protocolo institucional aplica."
        )

    result = state.get("tree_result")
    if not result:
        outcome = state.get("tree_outcome") or "ausencia_de_coincidencias"
        return _contribution(
            "arbol",
            outcome,
            detail=state.get("tree_stop_detail")
            or f"El recorrido del arbol del protocolo {state.get('tree_protocol_id')} no llego a un resultado.",
        )

    citations = result.get("citations") or []
    complete = [
        c
        for c in citations
        if c.get("source_file")
        and c.get("page") is not None
        and (c.get("verbatim_text") or "").strip()
    ]
    references = [
        {
            "protocol_id": result.get("protocol_id"),
            "protocol_version": result.get("protocol_version"),
            "rule_id": result.get("rule_id"),
            "source_file": c.get("source_file"),
            "section": c.get("section"),
            "page": c.get("page"),
        }
        for c in citations
    ]
    if not complete:
        return _contribution(
            "arbol",
            "respuesta_fundamentada",
            verification="sin_procedencia",
            content=result.get("answer_text", ""),
            references=references,
            detail=f"El nodo {result.get('rule_id')} no tiene citas completas al documento fuente.",
        )
    return _contribution(
        "arbol",
        "respuesta_fundamentada",
        verification="verificada",
        eligible=True,
        content=result.get("answer_text", ""),
        references=references,
    )


def _rag_reference(rc: RetrievedChunk) -> dict[str, Any]:
    return {
        "source_file": rc.chunk.source_file,
        "section": rc.chunk.section,
        "page": rc.chunk.page,
    }


def _verify_rag(state: AgentState) -> dict[str, Any]:
    if state.get("rag_error"):
        return _contribution("rag", "error_tecnico", detail=state.get("rag_error"))

    answer = state.get("answer", "") or ""
    sources: list[RetrievedChunk] = state.get("sources", []) or []
    if is_refusal(answer):
        return _contribution("rag", "abstencion", content=answer)
    if not sources:
        return _contribution(
            "rag", "ausencia_de_coincidencias", detail="No se recuperaron fragmentos."
        )

    status = state.get("eval_status") or (
        "aprobada" if state.get("eval_ok", True) else "rechazada"
    )
    if status == "no_verificable":
        return _contribution(
            "rag",
            "respuesta_fundamentada",
            verification="no_verificable",
            detail=state.get("eval_feedback"),
        )
    if status != "aprobada":
        return _contribution(
            "rag",
            "respuesta_fundamentada",
            verification="rechazada_por_evaluador",
            detail=state.get("eval_feedback"),
        )

    references = [
        _rag_reference(rc)
        for rc in sources
        if rc.chunk.source_file and rc.chunk.page is not None
    ]
    if not references:
        return _contribution(
            "rag",
            "respuesta_fundamentada",
            verification="sin_procedencia",
            detail="La respuesta no tiene fuentes con archivo y pagina.",
        )
    return _contribution(
        "rag",
        "respuesta_fundamentada",
        verification="verificada",
        eligible=True,
        content=answer,
        references=references,
    )


def _version_discrepancies(tree: dict[str, Any], rag: dict[str, Any]) -> list[str]:
    tree_versions = _protocol_versions(
        [r.get("source_file") or "" for r in tree["references"]]
    )
    rag_versions = _protocol_versions(
        [r.get("source_file") or "" for r in rag["references"]]
    )
    notes = []
    for code in sorted(set(tree_versions) & set(rag_versions)):
        if tree_versions[code] != rag_versions[code]:
            notes.append(
                f"El arbol de decision usa {code} version {', '.join(sorted(tree_versions[code]))} "
                f"y los documentos recuperados citan {code} version "
                f"{', '.join(sorted(rag_versions[code]))}."
            )
    return notes


_RELEVANCE_SYSTEM_PROMPT = """\
Eres un auditor. Decide si la APORTACION ARBOL (arbol clinico de un protocolo \
institucional) responde, total o parcialmente, a la PREGUNTA del usuario. \
No respondas la pregunta ni uses conocimiento propio.

Responde UNICAMENTE con un objeto JSON: {"relacionada": true} si la \
aportacion responde a lo que se pregunta, o {"relacionada": false} si trata \
otro tema o no aporta a la respuesta.
"""


def _tree_is_related(
    question: str, tree: dict[str, Any], cfg: DictConfig
) -> bool | None:
    """True/False segun el juez, o None si no fue posible decidirlo."""
    user_prompt = f"PREGUNTA:\n{question}\n\nAPORTACION ARBOL:\n{tree['content']}"
    try:
        client = Client(host=cfg.env.ollama_url, timeout=llm_timeout(cfg))
        for attempt in range(1, _CONTRADICTION_ATTEMPTS + 1):
            response = client.chat(
                model=cfg.agents.evaluator_model,
                messages=[
                    {"role": "system", "content": _RELEVANCE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                format=RELEVANCE_SCHEMA,
                options=ollama_options(cfg, 50),
                stream=False,
            )
            content = response["message"]["content"].strip()
            try:
                flag = json.loads(content).get("relacionada")
            except (json.JSONDecodeError, AttributeError):
                flag = None
            if isinstance(flag, bool):
                return flag
            logger.warning(
                "Verificador: veredicto de relacion no interpretable (intento %d/%d): %r",
                attempt,
                _CONTRADICTION_ATTEMPTS,
                content,
            )
    except Exception:  # un fallo del modelo no debe tumbar la verificacion
        logger.exception("Verificador: fallo la comprobacion de relacion del arbol.")
    return None


def _content_discrepancy(
    question: str, tree: dict[str, Any], rag: dict[str, Any], cfg: DictConfig
) -> str | None:
    """Nota de contradiccion entre arbol y RAG, o None si no se contradicen."""
    user_prompt = (
        f"PREGUNTA:\n{question}\n\n"
        f"APORTACION ARBOL:\n{tree['content']}\n\n"
        f"APORTACION RAG:\n{rag['content']}"
    )
    try:
        client = Client(host=cfg.env.ollama_url, timeout=llm_timeout(cfg))
        for attempt in range(1, _CONTRADICTION_ATTEMPTS + 1):
            response = client.chat(
                model=cfg.agents.evaluator_model,
                messages=[
                    {"role": "system", "content": _CONTRADICTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                format=CONTRADICTION_SCHEMA,
                options=ollama_options(cfg, 200),
                stream=False,
            )
            content = response["message"]["content"].strip()
            try:
                parsed = json.loads(content)
                flag = parsed.get("contradiccion")
            except (json.JSONDecodeError, AttributeError):
                flag = None
            if flag is False:
                return None
            if flag is True:
                detail = str(parsed.get("detalle", "")).strip()
                return (
                    "Posible contradiccion entre el arbol de decision y los documentos recuperados"
                    + (f": {detail}" if detail else ".")
                )
            logger.warning(
                "Verificador: veredicto de contradiccion no interpretable (intento %d/%d): %r",
                attempt,
                _CONTRADICTION_ATTEMPTS,
                content,
            )
    except Exception:  # un fallo del modelo no debe tumbar la verificacion
        logger.exception("Verificador: fallo la comprobacion de contradicciones.")
    return "No fue posible comprobar si el arbol de decision y los documentos recuperados se contradicen."


def verify_contributions(state: AgentState, cfg: DictConfig) -> dict:
    tree = _verify_tree(state)
    rag = _verify_rag(state)

    # La aportacion del arbol solo se publica si responde a la pregunta
    # (falla cerrada: si no se puede decidir, no se publica).
    if tree["eligible"]:
        related = _tree_is_related(state["question"], tree, cfg)
        if related is not True:
            tree = {
                **tree,
                "eligible": False,
                "verification": "no_relacionada"
                if related is False
                else "no_verificable",
                "detail": "La aportacion del arbol no responde a la pregunta."
                if related is False
                else "No fue posible comprobar si la aportacion del arbol responde a la pregunta.",
            }

    discrepancies: list[str] = []
    if tree["eligible"] and rag["eligible"]:
        discrepancies.extend(_version_discrepancies(tree, rag))
        note = _content_discrepancy(state["question"], tree, rag, cfg)
        if note:
            discrepancies.append(note)

    logger.info(
        "Verificador: arbol=%s/%s rag=%s/%s discrepancias=%d",
        tree["status"],
        tree["verification"],
        rag["status"],
        rag["verification"],
        len(discrepancies),
    )
    # Para diagnostico: el texto de las aportaciones que no se publican (el
    # chat no las muestra, por la falla cerrada).
    unpublished = {
        "arbol": tree["content"],
        # Las respuestas rechazadas por el evaluador no traen contenido en la
        # aportacion; se registra el texto que genero el RAG.
        "rag": rag["content"] or state.get("answer") or "",
    }
    for contribution in (tree, rag):
        text = unpublished[contribution["agent"]]
        if text and not contribution["eligible"]:
            logger.info(
                "Verificador: aportacion no publicada (%s): %r",
                contribution["agent"],
                text[:_LOG_SNIPPET_CHARS],
            )
    return {"contributions": [tree, rag], "discrepancies": discrepancies}
