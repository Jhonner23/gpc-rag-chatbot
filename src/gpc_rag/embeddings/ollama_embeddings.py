"""Cliente de embeddings usando un modelo servido por Ollama (ej. bge-m3)."""

from __future__ import annotations

import logging

from ollama import Client
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    """Envuelve el cliente de Ollama para generar embeddings en batch."""

    def __init__(self, cfg: DictConfig):
        self.model = cfg.embeddings_cfg.model
        self.client = Client(host=cfg.env.ollama_url)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Genera un embedding por cada texto de entrada.

        Ollama no siempre soporta batch real segun la version del servidor,
        asi que se hace una llamada por texto -- para indexar cientos de
        chunks esto es aceptable (proceso batch, no interactivo).
        """
        embeddings: list[list[float]] = []
        for text in texts:
            response = self.client.embed(model=self.model, input=text)
            vector = response["embeddings"][0]
            embeddings.append(vector)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]
