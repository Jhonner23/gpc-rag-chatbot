"""Chatbot (Chainlit) -- la interfaz que ven los usuarios finales.

Cliente HTTP delgado sobre la API FastAPI (pipeline RAG + agentes de la
seccion 9 de la arquitectura: coordinador, agente arbol, RAG, evaluador,
comparador).

El agente arbol puede pausar la conversacion para preguntar, una variable a
la vez, un dato que le falta (ver agents/tree_agent.py). Mientras eso pasa,
esta pantalla guarda el `thread_id` de esa pausa en la sesion de Chainlit; el
siguiente mensaje del usuario se manda como la RESPUESTA a esa pregunta (no
como una pregunta nueva), hasta que la API devuelva una respuesta final.

Correr local (con la API ya corriendo en :8000, GPC_USE_AGENTS=true para que
el wizard funcione):
    uv run chainlit run src/gpc_rag/chat/app.py --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import os

import chainlit as cl
import httpx

from gpc_rag.common.refusal import is_refusal

API_URL = os.environ.get("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 300
_ASSISTANT_AUTHOR = "Asistente GPC"
_PENDING_THREAD_KEY = "tree_wizard_thread_id"


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set(_PENDING_THREAD_KEY, None)
    await cl.Message(
        content=(
            "Hola, soy el asistente de Guias de Practica Clinica (GPC). "
            "Pregunta sobre el contenido de las guias indexadas y respondo citando "
            "la fuente exacta. Recuerda: esto apoya pero no reemplaza el juicio clinico."
        ),
        author=_ASSISTANT_AUTHOR,
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


async def _call_chat_api(payload: dict) -> dict | None:
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(f"{API_URL}/chat", json=payload)
            response.raise_for_status()
            data: dict = response.json()
            return data
    except httpx.HTTPError as exc:
        await cl.Message(
            content=(
                "No pude comunicarme con el servicio de RAG. "
                f"Verifica que la API este corriendo en {API_URL}. Detalle: {exc}"
            ),
            author=_ASSISTANT_AUTHOR,
        ).send()
        return None


@cl.on_message
async def on_message(message: cl.Message) -> None:
    text = message.content.strip()
    if not text:
        return

    pending_thread_id = cl.user_session.get(_PENDING_THREAD_KEY)
    if pending_thread_id:
        payload = {"thread_id": pending_thread_id, "answer": text}
    else:
        payload = {"question": text}

    thinking = cl.Message(content="Pensando...", author=_ASSISTANT_AUTHOR)
    await thinking.send()

    data = await _call_chat_api(payload)
    if data is None:
        await thinking.remove()
        return

    if data.get("pending_question"):
        cl.user_session.set(_PENDING_THREAD_KEY, data["thread_id"])
        thinking.content = data["pending_question"]
        await thinking.update()
        return

    cl.user_session.set(_PENDING_THREAD_KEY, None)
    answer_text = data["answer"]
    sources_block = "" if is_refusal(answer_text) else _format_sources(data.get("sources", []))
    thinking.content = answer_text + sources_block
    await thinking.update()
