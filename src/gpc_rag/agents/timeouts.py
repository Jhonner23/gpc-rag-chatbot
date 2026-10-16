"""Tiempos maximos de espera de la capa de agentes (ver conf/config.yaml).

- llm_timeout: limite de cada llamada a Ollama hecha por los agentes
  (coordinador, arbol, evaluador, verificador e integrador).
- rag_timeout: limite de la rama completa del RAG (recuperacion + generacion).
  Si se supera, la rama queda en estado "error_tecnico" y su resultado tardio
  se descarta: nunca se asocia a otra consulta.

Los valores por defecto dejan margen sobre lo medido en el equipo de
desarrollo (recuperacion con mediana de ~55 s y maximo de ~80 s).
"""

from __future__ import annotations

from omegaconf import DictConfig

DEFAULT_LLM_TIMEOUT_S = 120.0
DEFAULT_RAG_TIMEOUT_S = 240.0


def _agents_value(cfg: DictConfig, key: str, default: float) -> float:
    agents = cfg.get("agents") or {}
    value = agents.get(key) if hasattr(agents, "get") else None
    return float(value) if value else default


def llm_timeout(cfg: DictConfig) -> float:
    return _agents_value(cfg, "llm_timeout_s", DEFAULT_LLM_TIMEOUT_S)


def rag_timeout(cfg: DictConfig) -> float:
    return _agents_value(cfg, "rag_timeout_s", DEFAULT_RAG_TIMEOUT_S)
