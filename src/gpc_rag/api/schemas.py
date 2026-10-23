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
    source_type: str = "institucional"
    source_title: str = ""


class AgentPanelOut(BaseModel):
    """Aportacion de un agente (arbol, RAG), con su estado y sus fuentes."""

    agent: str
    title: str
    status: str
    eligible: bool
    content: str
    sources: list[SourceOut] = []


class ChatResponse(BaseModel):
    """Si `pending_question` viene con texto, la conversacion esta pausada
    esperando esa respuesta -- `answer`/`sources` vienen vacios, y hay que
    reenviar `thread_id` + la respuesta del usuario en el siguiente POST
    (como `answer` del ChatRequest) para continuar.

    Con la capa de agentes, `panels` trae la respuesta de cada agente por
    separado, `notes` las notas del verificador (contradicciones o versiones
    distintas entre fuentes) y `answer` la composicion literal de las
    aportaciones verificadas."""

    question: str
    answer: str | None = None
    sources: list[SourceOut] = []
    panels: list[AgentPanelOut] = []
    notes: list[str] = []
    pending_question: str | None = None
    thread_id: str | None = None
