"""Tipos de datos compartidos entre modulos (chunks, resultados de busqueda, etc)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """Un fragmento de una Guia de Practica Clinica, listo para indexar."""

    text: str
    source_file: str
    section: str = "Sin seccion"
    page: int | None = None
    chunk_index: int = 0
    chunk_id: str = ""
    # "institucional" (protocolos de la CUB), "guia" (guia externa) o
    # "revision" (articulo de revision o monografia). Ver ingestion/sources.py.
    source_type: str = "institucional"
    source_title: str = ""

    def metadata(self) -> dict:
        return {
            "source_file": self.source_file,
            "section": self.section,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "source_type": self.source_type,
            "source_title": self.source_title,
        }


@dataclass
class RetrievedChunk:
    """Un chunk recuperado en una busqueda, con su score."""

    chunk: Chunk
    score: float
    retrieval_method: str = "dense"  # "dense" | "bm25" | "hybrid" | "rerank"


@dataclass
class PendingTreeQuestion:
    """El grafo de agentes quedo pausado: el agente arbol necesita un dato
    mas (ver agents/tree_agent.py, agents/graph.py). `thread_id` identifica
    la conversacion pausada en el checkpointer -- hay que reenviarlo junto
    con la respuesta del usuario para continuar donde quedo."""

    thread_id: str
    question: str


def _sources_as_dicts(sources: list[RetrievedChunk]) -> list[dict]:
    return [
        {
            "source_file": rc.chunk.source_file,
            "section": rc.chunk.section,
            "page": rc.chunk.page,
            "score": round(rc.score, 4),
            "source_type": rc.chunk.source_type,
            "source_title": rc.chunk.source_title,
        }
        for rc in sources
    ]


@dataclass
class AgentPanel:
    """Lo que aporto un agente (arbol, RAG y, a futuro, casos similares),
    para mostrarlo por separado junto a la sugerencia consolidada. Solo trae
    contenido y fuentes si la aportacion fue verificada; si no, `content`
    explica por que el agente no llego a una respuesta publicable."""

    agent: str
    title: str
    status: str
    eligible: bool
    content: str
    sources: list[RetrievedChunk] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "agent": self.agent,
            "title": self.title,
            "status": self.status,
            "eligible": self.eligible,
            "content": self.content,
            "sources": _sources_as_dicts(self.sources),
        }


@dataclass
class RagAnswer:
    """Respuesta final del pipeline de RAG, lista para mostrar al usuario.

    `answer` es la composicion literal de las aportaciones verificadas (o la
    respuesta directa del RAG si no se usa la capa de agentes); `panels` trae
    la aportacion de cada agente por separado y `notes` las notas del
    verificador (ambos vacios fuera de la capa de agentes)."""

    question: str
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    panels: list[AgentPanel] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def sources_as_dicts(self) -> list[dict]:
        return _sources_as_dicts(self.sources)
