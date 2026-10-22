"""Registro de protocolos clinicos: descubre, carga y cachea los arboles
aprobados por el comite de la clinica (ver agents/flowchart.py).

Estructura esperada: agents/protocols/<protocol_id>/<version>/arbol.yaml

Agregar o actualizar un protocolo es copiar el YAML aprobado a esa carpeta,
sin tocar codigo. Si hay mas de una carpeta de version para el mismo
protocolo, se usa la que ordena ultimo alfabeticamente (p. ej. v10 > v09).

Se cachea porque `tree_agent_node` se reproduce completo en cada reanudacion
del wizard (interrupt()), y sin cache el YAML se volveria a leer en cada
pregunta.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from gpc_rag.agents.flowchart import Flowchart, load_flowchart

logger = logging.getLogger(__name__)

PROTOCOLS_DIR = Path(__file__).parent / "protocols"
TREE_FILENAME = "arbol.yaml"

_cache: dict[str, Flowchart] = {}
_lock = threading.Lock()


@dataclass(frozen=True)
class ProtocolInfo:
    id: str
    name: str
    description: str


def _discover_tree_files() -> dict[str, Path]:
    found: dict[str, Path] = {}
    if not PROTOCOLS_DIR.is_dir():
        return found
    for protocol_dir in sorted(PROTOCOLS_DIR.iterdir()):
        if not protocol_dir.is_dir():
            continue
        version_dirs = sorted(p for p in protocol_dir.iterdir() if p.is_dir())
        for version_dir in reversed(version_dirs):
            candidate = version_dir / TREE_FILENAME
            if candidate.is_file():
                found[protocol_dir.name] = candidate
                break
    return found


def get_flowchart(protocol_id: str) -> Flowchart:
    """Arbol del protocolo, cacheado tras la primera carga."""
    if protocol_id in _cache:
        return _cache[protocol_id]
    with _lock:
        if protocol_id not in _cache:
            path = _discover_tree_files().get(protocol_id)
            if path is None:
                raise KeyError(
                    f"No existe {TREE_FILENAME} para el protocolo {protocol_id!r} en {PROTOCOLS_DIR}"
                )
            chart = load_flowchart(path)
            if chart.protocol_id != protocol_id:
                raise ValueError(
                    f"{path} declara el protocolo {chart.protocol_id!r}, pero esta en la carpeta {protocol_id!r}."
                )
            _cache[protocol_id] = chart
            logger.info(
                "Registro de protocolos: cargado %s %s desde %s",
                protocol_id,
                chart.version,
                path,
            )
    return _cache[protocol_id]


def available_protocol_ids() -> list[str]:
    return sorted(_discover_tree_files().keys())


def describe_available_protocols() -> list[ProtocolInfo]:
    """Id, nombre y alcance de cada protocolo, para el prompt del clasificador."""
    infos = []
    for protocol_id in available_protocol_ids():
        chart = get_flowchart(protocol_id)
        infos.append(
            ProtocolInfo(
                id=chart.protocol_id, name=chart.name, description=chart.description
            )
        )
    return infos
