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

Falla cerrada (cambio posterior, para alinear el sistema con la
fundamentacion de la arquitectura multiagente): si el veredicto no es
interpretable se reintenta el juicio una vez, y si sigue sin serlo la
respuesta queda `no_verificable` y no se publica. Antes se aprobaba por
defecto.
"""

from __future__ import annotations

import json
import logging

from ollama import Client
from omegaconf import DictConfig

from gpc_rag.agents.schemas import EVALUATOR_SCHEMA
from gpc_rag.agents.timeouts import llm_timeout
from gpc_rag.common.grounding import grounding_problems
from gpc_rag.common.models import RetrievedChunk
from gpc_rag.common.refusal import is_refusal

logger = logging.getLogger(__name__)

_EVALUATOR_SYSTEM_PROMPT = """\
Eres un auditor que revisa si una respuesta clinica esta bien fundamentada \
en el CONTEXTO que se le dio al asistente que la genero. No respondas la \
pregunta tu mismo -- solo audita la respuesta dada.

Rechaza la respuesta si:
- Menciona un farmaco, dosis, esquema posologico o cifra especifica que NO \
aparece en el CONTEXTO. Unir en una frase datos de un mismo fragmento (p. ej. \
la dosis y la duracion de un farmaco en la misma fila de una tabla) o \
abreviar unidades ("h" por "horas") NO es un error.
- Cita una fuente que no tiene el formato \
[Guia: <archivo>, Seccion: <seccion>, Pagina: <pagina>], o que no \
corresponde a un fragmento del CONTEXTO (ej. CDC, OMS, articulos, URLs).
- Afirma tener informacion suficiente sobre un tema que el CONTEXTO no trata.
- Presenta como recomendacion algo que el CONTEXTO dice con otro alcance \
(p. ej. datos de frecuencia o de poblaciones de riesgo como si fueran una \
indicacion, una dosis de profilaxis como tratamiento), o atribuye una dosis \
o restriccion a un farmaco, poblacion o situacion distinta.
- Establece un orden de preferencia, una comparacion o una diferencia entre \
fuentes que el CONTEXTO no establece.

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


_JUDGE_ATTEMPTS = 2  # 1 intento + 1 reintento del juicio si el JSON no es interpretable

APROBADA = "aprobada"
RECHAZADA = "rechazada"
NO_VERIFICABLE = "no_verificable"


def evaluate_answer_status(
    question: str,
    sources: list[RetrievedChunk],
    answer: str,
    cfg: DictConfig,
) -> tuple[str, str | None]:
    """Devuelve (estado, motivo) con estado en {aprobada, rechazada, no_verificable}.

    Falla CERRADA: si el juez no devuelve un veredicto interpretable, se
    reintenta el juicio una vez; si vuelve a fallar, la respuesta queda como
    `no_verificable` y no se publica (antes se aprobaba por defecto). Un
    veredicto ilegible es un problema del juez, no de la respuesta, por eso
    `no_verificable` NO dispara la regeneracion del RAG (ver graph.py).
    """
    if is_refusal(answer):
        logger.info(
            "Evaluador: respuesta ya tiene forma de rechazo bien formado; se aprueba sin llamar al LLM."
        )
        return APROBADA, None

    problems = grounding_problems(answer, [rc.chunk.source_file for rc in sources])
    if problems:
        logger.info("Evaluador: rechazo determinista: %s", " ".join(problems))
        return RECHAZADA, " ".join(problems)

    client = Client(host=cfg.env.ollama_url, timeout=llm_timeout(cfg))
    user_prompt = (
        f"CONTEXTO:\n{_format_context(sources)}\n\n"
        f"PREGUNTA:\n{question}\n\n"
        f"RESPUESTA A AUDITAR:\n{answer}\n\n"
        "Recuerda: no respondas la pregunta. Devuelve solo el veredicto de la auditoria."
    )
    for attempt in range(1, _JUDGE_ATTEMPTS + 1):
        response = client.chat(
            model=cfg.agents.evaluator_model,
            messages=[
                {"role": "system", "content": _EVALUATOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            format=EVALUATOR_SCHEMA,
            options={"temperature": 0.0, "num_predict": 300},
            stream=False,
        )
        content = response["message"]["content"].strip()
        try:
            parsed = json.loads(content)
            verdict = str(parsed.get("verdict", "")).strip().upper()
        except (json.JSONDecodeError, AttributeError):
            logger.warning(
                "Evaluador: veredicto no interpretable (intento %d/%d): %r",
                attempt,
                _JUDGE_ATTEMPTS,
                content,
            )
            continue

        if verdict == "OK":
            return APROBADA, None
        if verdict == "RECHAZADA":
            reason = str(parsed.get("reason", "")).strip()
            return RECHAZADA, reason or "La respuesta no paso la auditoria."
        logger.warning(
            "Evaluador: veredicto inesperado (intento %d/%d): %r", attempt, _JUDGE_ATTEMPTS, verdict
        )

    logger.warning("Evaluador: sin veredicto interpretable; la respuesta queda como no verificable.")
    return NO_VERIFICABLE, "No fue posible verificar la respuesta contra el contexto recuperado."


def evaluate_answer(
    question: str,
    sources: list[RetrievedChunk],
    answer: str,
    cfg: DictConfig,
) -> tuple[bool, str | None]:
    """Compatibilidad: (aprobada, motivo). Solo `aprobada` cuenta como True."""
    status, reason = evaluate_answer_status(question, sources, answer, cfg)
    return status == APROBADA, reason
