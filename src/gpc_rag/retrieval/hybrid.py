"""Retrieval hibrido: busqueda densa (Qdrant) + BM25 (lexica), fusionadas con
Reciprocal Rank Fusion, y opcionalmente reordenadas con un cross-encoder.

El hibrido importa especialmente en GPC: muchas recomendaciones comparten
vocabulario muy similar (ej. "manejo farmacologico de X"), y la busqueda
puramente densa a veces no distingue bien terminos clinicos exactos que BM25
si captura (siglas, nombres de farmacos, codigos).
"""

from __future__ import annotations

import logging

from omegaconf import DictConfig
from rank_bm25 import BM25Okapi

from gpc_rag.common.models import Chunk, RetrievedChunk
from gpc_rag.embeddings.ollama_embeddings import OllamaEmbedder
from gpc_rag.vectorstore.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]], k: int = 60
) -> list[RetrievedChunk]:
    """Combina varias listas ordenadas en una sola, sin necesitar scores comparables."""
    scores: dict[str, float] = {}
    best_chunk: dict[str, RetrievedChunk] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            key = item.chunk.chunk_id
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in best_chunk:
                best_chunk[key] = item
    fused = [
        RetrievedChunk(chunk=best_chunk[key].chunk, score=score, retrieval_method="hybrid")
        for key, score in scores.items()
    ]
    fused.sort(key=lambda rc: rc.score, reverse=True)
    return fused


class HybridRetriever:
    def __init__(self, cfg: DictConfig, embedder: OllamaEmbedder, store: QdrantStore):
        self.cfg = cfg
        self.embedder = embedder
        self.store = store
        self._bm25: BM25Okapi | None = None
        self._bm25_chunks: list[Chunk] = []
        self._reranker = None  # se carga perezosamente, solo si se usa

    def build_bm25_index(self) -> None:
        """Reconstruye el indice BM25 en memoria a partir de lo que hay en Qdrant.

        Llamar despues de cada reindexacion (ver pipelines/feature_pipeline).
        Para corpus muy grandes esto se deberia persistir en disco en vez de
        reconstruirse en memoria en cada arranque; para el volumen tipico de
        GPC de una clinica (decenas/cientos de guias) es suficiente.
        """
        chunks = self.store.scroll_all_chunks()
        self._bm25_chunks = chunks
        tokenized = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None
        logger.info("Indice BM25 construido con %d chunks.", len(chunks))

    def _bm25_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self._bm25 is None or not self._bm25_chunks:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(
            zip(self._bm25_chunks, scores, strict=True), key=lambda x: x[1], reverse=True
        )[:top_k]
        return [
            RetrievedChunk(chunk=chunk, score=float(score), retrieval_method="bm25")
            for chunk, score in ranked
            if score > 0
        ]

    @property
    def reranker(self):
        # Import perezoso: evita cargar torch/sentence-transformers si no se usa reranking.
        if self._reranker is None:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415

            self._reranker = CrossEncoder(self.cfg.retrieval.reranker_model, device="cpu")
        return self._reranker

    def _rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [(query, rc.chunk.text) for rc in candidates]
        scores = self.reranker.predict(pairs)
        reranked = [
            RetrievedChunk(chunk=rc.chunk, score=float(score), retrieval_method="rerank")
            for rc, score in zip(candidates, scores, strict=True)
        ]
        reranked.sort(key=lambda rc: rc.score, reverse=True)
        return reranked[:top_k]

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        query_vector = self.embedder.embed_query(query)
        dense_results = self.store.search(query_vector, top_k=self.cfg.retrieval.top_k_dense)
        bm25_results = self._bm25_search(query, top_k=self.cfg.retrieval.top_k_bm25)

        fused = _reciprocal_rank_fusion([dense_results, bm25_results])

        if self.cfg.retrieval.use_reranker and fused:
            candidate_pool = fused[: max(self.cfg.retrieval.top_k_final * 3, 10)]
            return self._rerank(query, candidate_pool, self.cfg.retrieval.top_k_final)

        return fused[: self.cfg.retrieval.top_k_final]
