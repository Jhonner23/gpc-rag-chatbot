"""Opciones comunes de las llamadas a Ollama.

`num_ctx`: Ollama usa por defecto una ventana de contexto corta (2048 a 4096
tokens segun la version) y, si el prompt no cabe, lo recorta SIN error desde
el principio. El prompt del RAG (reglas del sistema + 8 fragmentos) supera
facilmente los 4000 tokens, asi que el modelo podia estar respondiendo sin
ver parte de las reglas o de los fragmentos. Se fija explicitamente
(conf/llm/*.yaml, llm.num_ctx).
"""

from __future__ import annotations

from omegaconf import DictConfig

DEFAULT_NUM_CTX = 8192


def num_ctx(cfg: DictConfig) -> int:
    llm = cfg.get("llm") if hasattr(cfg, "get") else None
    value = llm.get("num_ctx") if llm is not None and hasattr(llm, "get") else None
    return int(value) if value else DEFAULT_NUM_CTX


def ollama_options(cfg: DictConfig, num_predict: int, temperature: float = 0.0) -> dict:
    """Opciones de una llamada de juicio o clasificacion (deterministas)."""
    return {"temperature": temperature, "num_predict": num_predict, "num_ctx": num_ctx(cfg)}
