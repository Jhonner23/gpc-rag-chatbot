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

    def metadata(self) -> dict:
        return {
            "source_file": self.source_file,
            "section": self.section,
            "page": self.page,
            "chunk_index": self.chunk_index,
        }


@dataclass
class RetrievedChunk:
    """Un chunk recuperado en una busqueda, con su score."""

    chunk: Chunk
    score: float
    retrieval_method: str = "dense"  # "dense" | "bm25" | "hybrid" | "rerank"


@dataclass
class RagAnswer:
    """Respuesta final del pipeline de RAG, lista para mostrar al usuario."""

    question: str
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)

    def sources_as_dicts(self) -> list[dict]:
        return [
            {
                "source_file": rc.chunk.source_file,
                "section": rc.chunk.section,
                "page": rc.chunk.page,
                "score": round(rc.score, 4),
            }
            for rc in self.sources
        ]
