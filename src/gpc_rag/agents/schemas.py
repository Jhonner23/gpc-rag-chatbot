"""Esquemas JSON para las salidas estructuradas de Ollama.

Con `format="json"` Ollama solo garantiza que la salida sea JSON valido, no
que tenga la forma pedida: en pruebas reales el juez de 7B respondio la
pregunta del usuario dentro del JSON en vez de emitir un veredicto, y el
texto se corto por el limite de tokens. Pasando un esquema JSON en `format`,
Ollama restringe la decodificacion a ese esquema (salidas estructuradas,
disponibles desde Ollama 0.5), de modo que el juez solo puede responder con
los campos y valores permitidos.
"""

from __future__ import annotations

from typing import Any


def verdict_schema(*allowed: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": list(allowed)},
            "reason": {"type": "string"},
        },
        "required": ["verdict"],
    }


FILTER_SCHEMA = verdict_schema("ADECUADA", "RECHAZADA")
EVALUATOR_SCHEMA = verdict_schema("OK", "RECHAZADA")
FIDELITY_SCHEMA = verdict_schema("OK", "RECHAZADA")

CONTRADICTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "contradiccion": {"type": "boolean"},
        "detalle": {"type": "string"},
    },
    "required": ["contradiccion"],
}

RELEVANCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"relacionada": {"type": "boolean"}},
    "required": ["relacionada"],
}


def protocol_schema(protocol_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "protocol": {"type": "string", "enum": [*protocol_ids, "NINGUNO"]},
            "modo": {"type": "string", "enum": ["caso", "consulta"]},
        },
        "required": ["protocol", "modo"],
    }


def decisions_schema(options_by_decision: dict[str, list[str]], unknown: str) -> dict[str, Any]:
    """Una propiedad por decision del arbol, restringida a sus opciones o a
    `unknown`."""
    return {
        "type": "object",
        "properties": {
            decision_id: {"type": "string", "enum": [*options, unknown]}
            for decision_id, options in options_by_decision.items()
        },
        "required": list(options_by_decision),
    }
