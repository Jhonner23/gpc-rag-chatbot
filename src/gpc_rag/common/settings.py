"""Carga de configuracion (Hydra/OmegaConf) para todo el proyecto.

Uso:
    from gpc_rag.common.settings import load_config
    cfg = load_config()              # usa la variable de entorno RAG_ENV (default: "dev")
    cfg = load_config(env="docker")  # fuerza el ambiente

La configuracion vive en `conf/` (raiz del repo). Este modulo la ubica de forma
relativa a este archivo para que funcione igual corriendo local o dentro de Docker.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONF_DIR = _REPO_ROOT / "conf"


@lru_cache(maxsize=8)
def load_config(env: str | None = None) -> DictConfig:
    """Compone la configuracion de `conf/config.yml` con el grupo `env` indicado.

    Args:
        env: "dev" o "docker". Si no se pasa, se lee de la variable de entorno
            RAG_ENV, y si tampoco existe, se usa "dev".

    Returns:
        DictConfig con toda la configuracion resuelta (llm, embeddings_cfg,
        vectorstore_cfg, chunking, retrieval, env, app).
    """
    resolved_env = env or os.environ.get("RAG_ENV", "dev")
    if not _CONF_DIR.exists():
        raise FileNotFoundError(
            f"No se encontro el directorio de configuracion en {_CONF_DIR}. "
            "Verifica que estas corriendo desde la raiz del repo."
        )
    with initialize_config_dir(version_base=None, config_dir=str(_CONF_DIR)):
        return compose(config_name="config", overrides=[f"env={resolved_env}"])
