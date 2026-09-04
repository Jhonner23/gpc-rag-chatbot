"""Pipeline de consulta (inference pipeline): pregunta -> retrieval -> respuesta.

Es la pieza que usan tanto la API (FastAPI) como, indirectamente via HTTP, el
chatbot (Chainlit). Se mantiene como un objeto reutilizable para no recargar
modelos/clientes en cada pregunta.
"""

from __future__ import annotations

from omegaconf import DictConfig

from gpc_rag.common.models import RagAnswer
from gpc_rag.common.settings import load_config
from gpc_rag.embeddings.ollama_embeddings import OllamaEmbedder
from gpc_rag.generation.ollama_llm import OllamaGenerator
from gpc_rag.retrieval.hybrid import HybridRetriever
from gpc_rag.vectorstore.qdrant_store import QdrantStore


class QueryPipeline:
    def __init__(self, cfg: DictConfig | None = None):
        self.cfg = cfg or load_config()
        self.embedder = OllamaEmbedder(self.cfg)
        # vector_size solo importa para crear la coleccion; en consulta ya existe.
        self.store = QdrantStore(self.cfg, vector_size=self.cfg.embeddings_cfg.dimensions)
        self.retriever = HybridRetriever(self.cfg, self.embedder, self.store)
        self.retriever.build_bm25_index()
        self.generator = OllamaGenerator(self.cfg)

    def answer(self, question: str) -> RagAnswer:
        retrieved = self.retriever.retrieve(question)
        answer_text = self.generator.generate(question, retrieved)
        return RagAnswer(question=question, answer=answer_text, sources=retrieved)


_pipeline_singleton: QueryPipeline | None = None


def get_pipeline() -> QueryPipeline:
    """Devuelve una unica instancia del pipeline (carga perezosa, cachea el reranker/BM25)."""
    global _pipeline_singleton  # noqa: PLW0603 -- singleton simple para cachear reranker/BM25 en memoria
    if _pipeline_singleton is None:
        _pipeline_singleton = QueryPipeline()
    return _pipeline_singleton
