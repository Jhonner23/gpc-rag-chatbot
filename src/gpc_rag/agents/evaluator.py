"""Agente evaluador: guardrail EN LINEA que audita la respuesta del agente
RAG contra el contexto recuperado, ANTES de que llegue al usuario.

Distinto de la evaluacion offline con RAGAS (ver scripts/run_ragas_eval.py):
RAGAS mide calidad agregada sobre un dataset, despues de los hechos, y no
bloquea nada en produccion. Este agente corre en cada pregunta real y puede
forzar un reintento o bloquear la respuesta (ver agents/graph.py).

Dos mitigaciones sobre la version inicial (ver diagnostico en la sesion: el
modelo evaluador generaba parrafos libres en vez de un veredicto corto, y
hasta fabricaba citas externas inexistentes):

1. Modo JSON nativo de Ollama (`format="json"`) en vez de pedir un formato de
   texto libre -- igual que en coordinator.py.
2. Short-circuit: si la respuesta del agente RAG ya tiene forma de "rechazo"
   bien formado (frases como "no encontre informacion..."), se aprueba
   directamente sin llamar al LLM evaluador. Un rechazo asi ya es, por
   definicion, la respuesta correcta cuando las guias no cubren el tema, y
   el heuristico de texto (common/refusal.py) ha demostrado ser mucho mas
   confiable que el juicio del LLM local para este caso puntual.
"""

from __future__ import annotations

import json
import logging

from ollama import Client
from omegaconf import DictConfig

from gpc_rag.common.models import RetrievedChunk
from gpc_rag.common.refusal import is_refusal

logger = logging.getLogger(__name__)

_EVALUATOR_SYSTEM_PROMPT = """\
Eres un auditor que revisa si una respuesta clinica esta bien fundamentada \
en el CONTEXTO que se le dio al asistente que la genero. No respondas la \
pregunta tu mismo -- solo audita la respuesta dada.

Rechaza la respuesta si:
- Menciona un farmaco, dosis, esquema posologico o cifra especifica que NO \
aparece textualmente en el CONTEXTO.
- Cita una fuente que no tiene el formato \
[Guia: <archivo>, Seccion: <seccion>, Pagina: <pagina>], o que no \
corresponde a un fragmento del CONTEXTO (ej. CDC, OMS, articulos, URLs).
- Afirma tener informacion suficiente sobre un tema que el CONTEXTO no trata.

Aprueba la respuesta si se ciñe al CONTEXTO, o si reconoce explicitamente \
que no hay informacion suficiente / que el tema no esta cubierto por las \
guias -- esa es una respuesta correcta, no una alucinacion.

Responde UNICAMENTE con un objeto JSON, sin texto adicional, con esta forma \
exacta:
- Si se aprueba: {"verdict": "OK"}
- Si se rechaza: {"verdict": "RECHAZADA", "reason": "<motivo breve y \
especifico, en español>"}
"""


def _format_context(sources: list[RetrievedChunk]) -> str:
    blocks = [
        f"[Guia: {rc.chunk.source_file}, Seccion: {rc.chunk.section}, "
        f"Pagina: {rc.chunk.page}]\n{rc.chunk.text}"
        for rc in sources
    ]
    return "\n\n".join(blocks) if blocks else "(sin contexto recuperado)"


def evaluate_answer(
    question: str,
    sources: list[RetrievedChunk],
    answer: str,
    cfg: DictConfig,
) -> tuple[bool, str | None]:
    """Devuelve (aprobada, motivo_de_rechazo_o_None)."""
    if is_refusal(answer):
        logger.info("Evaluador: respuesta ya tiene forma de rechazo bien formado; se aprueba sin llamar al LLM.")
        return True, None

    client = Client(host=cfg.env.ollama_url)
    user_prompt = (
        f"CONTEXTO:\n{_format_context(sources)}\n\n"
        f"PREGUNTA:\n{question}\n\n"
        f"RESPUESTA A AUDITAR:\n{answer}"
    )
    response = client.chat(
        model=cfg.agents.evaluator_model,
        messages=[
            {"role": "system", "content": _EVALUATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        format="json",
        options={"temperature": 0.0, "num_predict": 200},
        stream=False,
    )
    content = response["message"]["content"].strip()

    try:
        parsed = json.loads(content)
        verdict = str(parsed.get("verdict", "")).strip().upper()
    except (json.JSONDecodeError, AttributeError):
        logger.warning(
            "Evaluador: no se pudo parsear el JSON del veredicto (%r); se aprueba por defecto.",
            content,
        )
        return True, None

    if verdict == "OK":
        return True, None
    if verdict == "RECHAZADA":
        reason = str(parsed.get("reason", "")).strip()
        return False, reason or "La respuesta no paso la auditoria."

    logger.warning(
        "Evaluador: veredicto inesperado (%r); se aprueba por defecto.", verdict
    )
    return True, None
