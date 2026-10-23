"""Cliente de Qdrant: crear coleccion, indexar chunks y buscar por similitud."""

from __future__ import annotations

import logging

from omegaconf import DictConfig
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from gpc_rag.common.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


def _chunk_from_payload(payload: dict, point_id: object) -> Chunk:
    return Chunk(
        text=payload.get("text", ""),
        source_file=payload.get("source_file", "desconocido"),
        section=payload.get("section", "Sin seccion"),
        page=payload.get("page"),
        chunk_index=payload.get("chunk_index", 0),
        chunk_id=str(point_id),
        # Colecciones indexadas antes de que existieran las fuentes externas.
        source_type=payload.get("source_type", "institucional"),
        source_title=payload.get("source_title", ""),
    )


class QdrantStore:
    def __init__(self, cfg: DictConfig, vector_size: int):
        self.client = QdrantClient(url=cfg.env.qdrant_url)
        self.collection_name = cfg.app.collection_name
        self.vector_size = vector_size
        self.distance = cfg.vectorstore_cfg.distance

    def ensure_collection(self, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection_name)
        if exists and not recreate:
            return
        distance_map = {
            "cosine": qmodels.Distance.COSINE,
            "dot": qmodels.Distance.DOT,
            "euclid": qmodels.Distance.EUCLID,
        }
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=self.vector_size,
                distance=distance_map.get(self.distance, qmodels.Distance.COSINE),
            ),
        )
        logger.info("Coleccion '%s' creada en Qdrant.", self.collection_name)

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        points = [
            qmodels.PointStruct(
                id=chunk.chunk_id,
                vector=vector,
                payload={"text": chunk.text, **chunk.metadata()},
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(
        self, query_vector: list[float], top_k: int, source_type: str | None = None
    ) -> list[RetrievedChunk]:
        """Busqueda por similitud; con `source_type`, solo en ese tipo de fuente."""
        query_filter = (
            qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="source_type", match=qmodels.MatchValue(value=source_type)
                    )
                ]
            )
            if source_type
            else None
        )
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
        ).points
        return [
            RetrievedChunk(
                chunk=_chunk_from_payload(hit.payload or {}, hit.id),
                score=hit.score,
                retrieval_method="dense",
            )
            for hit in hits
        ]

    def scroll_all_chunks(self) -> list[Chunk]:
        """Trae todos los chunks indexados (para construir el indice BM25 en memoria)."""
        chunks: list[Chunk] = []
        next_offset = None
        while True:
            records, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            chunks.extend(
                _chunk_from_payload(record.payload or {}, record.id) for record in records
            )
            if next_offset is None:
                break
        return chunks
