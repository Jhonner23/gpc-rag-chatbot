"""Chatbot (Chainlit) -- la interfaz que ven los usuarios finales.

Dos modos separados, elegidos por el usuario al abrir el chat (chat profile),
nunca mezclados en el mismo mensaje -- ver seccion 10 de la arquitectura:

- "Chat libre": cliente HTTP delgado sobre la API FastAPI (pipeline RAG +
  agentes de la seccion 9). No cambio de logica respecto a la version
  anterior de este archivo.
- "Evaluacion guiada": wizard sobre un arbol de decision (trees/), 100%
  determinista y sin llamar a la API ni a ningun LLM -- ver chat/wizard.py.

En Docker `chat` y `api` siguen siendo dos servicios independientes; el modo
wizard no depende de la API porque el motor del arbol corre localmente en el
propio proceso de Chainlit.

Correr local (con la API ya corriendo en :8000):
    uv run chainlit run src/gpc_rag/chat/app.py --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import os

import chainlit as cl
import httpx

from gpc_rag.chat import wizard
from gpc_rag.common.refusal import is_refusal

API_URL = os.environ.get("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 300

_MODE_RAG = "Chat libre"
_MODE_WIZARD = "Evaluación guiada"


@cl.set_chat_profiles
async def chat_profiles() -> list[cl.ChatProfile]:
    return [
        cl.ChatProfile(
            name=_MODE_RAG,
            markdown_description=(
                "Pregunta lo que quieras sobre el contenido de las guias indexadas. "
                "Responde un modelo de lenguaje citando la fuente exacta."
            ),
        ),
        cl.ChatProfile(
            name=_MODE_WIZARD,
            markdown_description=(
                "Evaluacion paso a paso con un arbol de decision (ej. CURB-65, "
                "criterios IDSA/ATS de UCI). Sin modelo de lenguaje: la navegacion "
                "es determinista y cada resultado cita la pagina exacta de la guia."
            ),
        ),
    ]


@cl.on_chat_start
async def on_chat_start() -> None:
    profile = cl.user_session.get("chat_profile")

    if profile == _MODE_WIZARD:
        await wizard.start()
        return

    await cl.Message(
        content=(
            "Hola, soy el asistente de Guias de Practica Clinica (GPC). "
            "Pregunta sobre el contenido de las guias indexadas y respondo citando "
            "la fuente exacta. Recuerda: esto apoya pero no reemplaza el juicio clinico."
        )
    ).send()


def _format_sources(sources: list[dict]) -> str:
    if not sources:
        return ""
    lines = [
        f"- **{s['source_file']}** — Sección: {s['section']}"
        + (f", Pág. {s['page']}" if s.get("page") is not None else "")
        for s in sources
    ]
    return "\n\n**Fuentes consultadas:**\n" + "\n".join(lines)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    profile = cl.user_session.get("chat_profile")

    if profile == _MODE_WIZARD:
        # El unico texto libre que espera el wizard es la respuesta a una
        # pregunta numerica (las booleanas/enum se responden con botones).
        # Si no hay ninguna pregunta numerica pendiente, se ignora el mensaje
        # en vez de reinterpretarlo como pregunta libre -- este modo no habla
        # con el RAG.
        manejado = await wizard.handle_text_message(message)
        if not manejado:
            await cl.Message(
                content="Usa los botones de arriba para responder, o elige un protocolo del menu."
            ).send()
        return

    question = message.content.strip()
    if not question:
        return

    thinking = cl.Message(content="Buscando en las guias...")
    await thinking.send()

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(f"{API_URL}/chat", json={"question": question})
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        thinking.content = (
            "No pude comunicarme con el servicio de RAG. "
            f"Verifica que la API este corriendo en {API_URL}. Detalle: {exc}"
        )
        await thinking.update()
        return

    answer_text = data["answer"]
    sources_block = "" if is_refusal(answer_text) else _format_sources(data.get("sources", []))
    thinking.content = answer_text + sources_block
    await thinking.update()
