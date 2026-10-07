from __future__ import annotations

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Una pregunta nueva (`question`), o la respuesta del usuario a una
    pregunta pendiente del agente arbol (`thread_id` + `answer`, ver
    agents/tree_agent.py / agents/graph.py). No se mandan los dos a la vez:
    el chat (chat/app.py) decide cual segun si hay un wizard en curso."""

    question: str = ""
    thread_id: str | None = None
    answer: str | None = None


class SourceOut(BaseModel):
    source_file: str
    section: str
    page: int | None = None
    score: float


class ChatResponse(BaseModel):
    """Si `pending_question` viene con texto, la conversacion esta pausada
    esperando esa respuesta -- `answer`/`sources` vienen vacios, y hay que
    reenviar `thread_id` + la respuesta del usuario en el siguiente POST
    (como `answer` del ChatRequest) para continuar."""

    question: str
    answer: str | None = None
    sources: list[SourceOut] = []
    pending_question: str | None = None
    thread_id: str | None = None
