"""Cliente de Ollama con un reintento ante fallos transitorios del servidor.

Ollama ejecuta cada modelo en un proceso aparte ("runner"). Si ese proceso se
reinicia (al recargar el modelo o liberar memoria mientras el reranker ocupa
la CPU), la peticion que llega en ese momento falla con un 500, p. ej.
'Post "http://127.0.0.1:35611/tokenize": connect: connection refused'. Paso
con el clasificador del agente arbol: la rama quedo en "error tecnico"
aunque Ollama respondio bien a la siguiente peticion. Un reintento tras una
pausa corta lo cubre; los errores 4xx (peticion invalida) no se reintentan.
"""

from __future__ import annotations

import logging
import time

from ollama import Client as _OllamaClient
from ollama import ResponseError

logger = logging.getLogger(__name__)

RETRY_DELAY_SECONDS = 2.0


class Client(_OllamaClient):
    def chat(self, *args, **kwargs):
        if kwargs.get("stream"):
            return super().chat(*args, **kwargs)
        try:
            return super().chat(*args, **kwargs)
        except ResponseError as exc:
            if exc.status_code < 500:
                raise
            logger.warning("Ollama: error %s (%s); se reintenta una vez.", exc.status_code, exc.error)
            time.sleep(RETRY_DELAY_SECONDS)
            return super().chat(*args, **kwargs)
