"""Cliente del LLM de generacion, servido por Ollama."""

from __future__ import annotations

from collections.abc import Iterator

from ollama import Client
from omegaconf import DictConfig

from gpc_rag.common.models import RetrievedChunk
from gpc_rag.generation.prompts import build_messages


class OllamaGenerator:
    def __init__(self, cfg: DictConfig):
        self.model = cfg.llm.model
        self.temperature = cfg.llm.temperature
        self.max_tokens = cfg.llm.max_tokens
        self.client = Client(host=cfg.env.ollama_url)

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        messages = build_messages(question, chunks)
        response = self.client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": self.temperature, "num_predict": self.max_tokens},
            stream=False,
        )
        return response["message"]["content"]

    def generate_stream(self, question: str, chunks: list[RetrievedChunk]) -> Iterator[str]:
        """Version con streaming, para que el chatbot muestre la respuesta token a token."""
        messages = build_messages(question, chunks)
        for part in self.client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": self.temperature, "num_predict": self.max_tokens},
            stream=True,
        ):
            content = part.get("message", {}).get("content", "")
            if content:
                yield content
