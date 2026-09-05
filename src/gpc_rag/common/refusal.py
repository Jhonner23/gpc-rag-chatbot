"""Deteccion de respuestas de "rechazo" (no hay informacion suficiente / tema
no cubierto) a partir del texto de la respuesta.

Centralizado aca porque se usa en dos lugares que deben comportarse igual:
- chat/app.py: para decidir si mostrar el bloque "Fuentes consultadas".
- agents/evaluator.py: como short-circuit -- si la respuesta del agente RAG
  ya tiene forma de rechazo bien formado, se aprueba directamente sin gastar
  una llamada al LLM evaluador (que ademas ha demostrado ser poco confiable
  para juzgar exactamente este caso, ver commit del agente evaluador).
"""

from __future__ import annotations

REFUSAL_MARKERS = [
    "no encontr",
    "no cubr",
    "no teng",
    "no dispon",
    "informacion suficiente",
    "información suficiente",
]


def is_refusal(answer: str) -> bool:
    lower = answer.lower()
    return any(marker in lower for marker in REFUSAL_MARKERS)
