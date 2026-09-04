"""Chatbot (Chainlit) -- la interfaz que ven los usuarios finales.

Es un cliente HTTP delgado sobre la API FastAPI (no llama al pipeline
directamente), asi en Docker `chat` y `api` son dos servicios independientes
que se pueden escalar/reiniciar por separado.

Correr local (con la API ya corriendo en :8000):
    uv run chainlit run src/gpc_rag/chat/app.py --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import os

import chainlit as cl
import httpx

API_URL = os.environ.get("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 300


@cl.on_chat_start
async def on_chat_start() -> None:
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

    content = data["answer"] + _format_sources(data.get("sources", []))
    thinking.content = content
    await thinking.update()
