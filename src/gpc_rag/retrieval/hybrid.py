"""Retrieval hibrido: busqueda densa (Qdrant) + BM25 (lexica), fusionadas con
Reciprocal Rank Fusion, y opcionalmente reordenadas con un cross-encoder.

El hibrido importa especialmente en GPC: muchas recomendaciones comparten
vocabulario muy similar (ej. "manejo farmacologico de X"), y la busqueda
puramente densa a veces no distingue bien terminos clinicos exactos que BM25
si captura (siglas, nombres de farmacos, codigos).
"""

from __future__ import annotations

import logging
import re
import unicodedata

from omegaconf import DictConfig
from rank_bm25 import BM25Okapi

from gpc_rag.common.models import Chunk, RetrievedChunk
from gpc_rag.embeddings.ollama_embeddings import OllamaEmbedder
from gpc_rag.vectorstore.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Palabras en minuscula, sin tildes ni puntuacion.

    Antes se separaba solo por espacios, asi que "asintomática**", "ITU?" o
    "días," no coincidian con "asintomática", "ITU" o "días" de la pregunta:
    BM25 no encontraba, por ejemplo, la fila de bacteriuria asintomatica de la
    Tabla 1 del protocolo de ITU.
    """
    text = unicodedata.normalize("NFKD", (text or "").lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [w for w in re.findall(r"[a-z0-9]+", text) if w not in _STOPWORDS]


# Palabras vacias del español (sin tildes, como quedan tras _tokenize). Sin
# filtrarlas, en una pregunta como "¿En qué situaciones ... debe ser tratada
# y en cuáles no?" pesan "en", "que", "debe", "ser", "no", y BM25 favorece
# fragmentos largos que las repiten. "no" se conserva: en clinica cambia el
# sentido ("no tratar").
_STOPWORDS = frozenset(
    """a al algo algunas algunos ante antes como con contra cual cuales cuando de del
    desde donde durante e el ella ellas ellos en entre era es esa esas ese eso esos
    esta estan estas este esto estos fue fueron ha han hay la las le les lo los mas
    me mi muy nos o otra otras otro otros para pero por porque puede pueden que se
    segun ser si sin sobre son su sus tambien tiene tienen todo todos tu un una unas
    uno unos y ya debe deben cuyo cuya""".split()  # noqa: SIM905 -- lista legible
)


_RECOMMENDATION_SECTION_RE = re.compile(r"^\d{1,3}\.\s")
# Desactivado por defecto (retrieval.section_boost en conf/config.yaml): se
# diseno para la GPC de NAC 2026, que numeraba sus preguntas clinicas
# ("13. En pacientes con..."). Esa guia ya no esta en el corpus, y en los
# protocolos institucionales lo numerado son secciones genericas ("1.
# OBJETIVO"), que con el impulso desplazaban a fragmentos relevantes.
_DEFAULT_SECTION_BOOST = 0.0
# El boost solo se aplica si el reranker YA le dio a este chunk una senal
# minima de relevancia real. Sin este piso, el boost termina inventando
# relevancia de la nada: para una pregunta totalmente fuera de tema (ej.
# tuberculosis, cuando el corpus solo tiene NAC), TODOS los chunks tienen
# score real cercano a 0 -- y si igual boosteamos las secciones numeradas
# "N. Pregunta clinica...", terminan ganando el top-k solo por su formato,
# no por ser relevantes, lo que le da al LLM contexto fuera de tema y lo
# empuja a alucinar (responder con conocimiento propio) en vez de decir que
# no hay informacion suficiente. Este piso evita ese efecto.
_MIN_SCORE_FOR_BOOST = 0.02


def _recommendation_boost(section: str, raw_score: float, boost: float = _DEFAULT_SECTION_BOOST) -> float:
    """Impulso heuristico para chunks de secciones tipo "N. Pregunta clinica...".

    En una GPC, esas secciones contienen la Recomendacion formal (grado de
    recomendacion, calidad de evidencia, farmaco/dosis exacta) -- el contenido
    normativo que un clinico necesita citar. Un reranker generico (no
    entrenado en dominio clinico) a veces prefiere parrafos narrativos de
    evidencia/discusion, que mencionan mas terminos de la pregunta pero no dan
    la recomendacion en si. Este boost corrige ese sesgo sin descartar al
    reranker: solo suma un margen fijo, no reemplaza su score, y solo cuando
    el reranker ya considero el chunk minimamente relevante (ver
    _MIN_SCORE_FOR_BOOST) -- nunca sobre un chunk que el reranker considero
    irrelevante.
    """
    if raw_score < _MIN_SCORE_FOR_BOOST:
        return 0.0
    return boost if _RECOMMENDATION_SECTION_RE.match(section) else 0.0


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

            # max_length acota el costo en CPU: el modelo admite hasta 8192
            # tokens por par, y los fragmentos son de ~400 (ver chunking).
            self._reranker = CrossEncoder(
                self.cfg.retrieval.reranker_model,
                device="cpu",
                max_length=self.cfg.retrieval.get("reranker_max_length", 1024),
            )
        return self._reranker

    def _rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [(query, rc.chunk.text) for rc in candidates]
        raw_scores = [float(s) for s in self.reranker.predict(pairs)]
        reranked = [
            RetrievedChunk(
                chunk=rc.chunk,
                score=raw_score
                + _recommendation_boost(
                    rc.chunk.section,
                    raw_score,
                    self.cfg.retrieval.get("section_boost", _DEFAULT_SECTION_BOOST),
                ),
                retrieval_method="rerank",
            )
            for rc, raw_score in zip(candidates, raw_scores, strict=True)
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
