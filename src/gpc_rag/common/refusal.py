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


# Un rechazo bien formado es corto: la frase de rechazo y, a lo sumo, una
# aclaracion de que temas cubren las guias. Antes bastaba con que el marcador
# apareciera en cualquier parte, y una respuesta larga que deducia un
# tratamiento ("se puede inferir... probablemente incluiria un antibiotico
# que cubra S. pneumoniae") y terminaba con "No encuentro informacion
# suficiente... para una dosis" se trataba como abstencion: el evaluador la
# aprobaba sin revisarla y el chat la mostraba. Ahora pasa por el evaluador.
MAX_REFUSAL_WORDS = 60


def is_refusal(answer: str) -> bool:
    lower = (answer or "").lower()
    if len(lower.split()) > MAX_REFUSAL_WORDS:
        return False
    return any(marker in lower for marker in REFUSAL_MARKERS)
