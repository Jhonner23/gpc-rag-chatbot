"""Cliente del LLM de generacion, servido por Ollama."""

from __future__ import annotations

from collections.abc import Iterator

from ollama import Client
from omegaconf import DictConfig

from gpc_rag.common.llm_options import num_ctx
from gpc_rag.common.models import RetrievedChunk
from gpc_rag.generation.prompts import build_messages


class OllamaGenerator:
    def __init__(self, cfg: DictConfig):
        self.model = cfg.llm.model
        self.temperature = cfg.llm.temperature
        self.max_tokens = cfg.llm.max_tokens
        self.seed = cfg.llm.get("seed")
        self.num_ctx = num_ctx(cfg)
        self.client = Client(host=cfg.env.ollama_url)

    def _options(self) -> dict:
        options = {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
            "num_ctx": self.num_ctx,
        }
        if self.seed is not None:
            options["seed"] = self.seed
        return options

    def generate(
        self, question: str, chunks: list[RetrievedChunk], extra_instruction: str | None = None
    ) -> str:
        messages = build_messages(question, chunks, extra_instruction=extra_instruction)
        response = self.client.chat(
            model=self.model,
            messages=messages,
            options=self._options(),
            stream=False,
        )
        return response["message"]["content"]

    def chat_json(self, system: str, user: str, schema: dict, num_predict: int) -> str:
        """Llamada con salida JSON restringida a `schema` (modo extractivo)."""
        options = {**self._options(), "num_predict": num_predict}
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            format=schema,
            options=options,
            stream=False,
        )
        return response["message"]["content"]

    def generate_stream(self, question: str, chunks: list[RetrievedChunk]) -> Iterator[str]:
        """Version con streaming, para que el chatbot muestre la respuesta token a token."""
        messages = build_messages(question, chunks)
        for part in self.client.chat(
            model=self.model,
            messages=messages,
            options=self._options(),
            stream=True,
        ):
            content = part.get("message", {}).get("content", "")
            if content:
                yield content
