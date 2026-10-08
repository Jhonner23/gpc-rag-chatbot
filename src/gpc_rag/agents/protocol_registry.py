"""Registro de protocolos clinicos: descubre, carga y cachea los
ProtocolVersion a partir de los `package.yaml` vendorizados en este repo
(ver agents/protocols/<protocol_id>/<version>/package.yaml).

Por que vendorizar el YAML en vez de depender de las funciones
`build_*_package()` del paquete `cpg_tree` del compañero: esas funciones son
codigo Python especifico por protocolo -- cada protocolo nuevo exigiria que
el, o nosotros, escribamos una funcion mas y la registremos a mano.
`load_package()` es la capa generica de `cpg_tree` (no especifica a ningun
protocolo): agregar un protocolo nuevo pasa a ser copiar un `package.yaml`
mas a esta carpeta, sin tocar codigo.

Por que cachear: `load_package()` parsea el YAML en cada llamada (~96ms
medido empiricamente, ~150x mas lento que el `build_nac_package()` anterior
que solo construye objetos Python ya en memoria). `tree_agent_node` llama a
esto en cada pregunta del wizard dentro de una misma conversacion (LangGraph
reproduce el nodo completo en cada reanudacion), asi que sin cache esa
lentitud se pagaria una vez por pregunta. Con el cache, se paga una sola vez
por protocolo por proceso.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from cpg_tree.knowledge import load_package
from cpg_tree.knowledge.protocol import Protocol, ProtocolVersion

logger = logging.getLogger(__name__)

PROTOCOLS_DIR = Path(__file__).parent / "protocols"

_cache: dict[str, ProtocolVersion] = {}
_lock = threading.Lock()


def _discover_package_files() -> dict[str, Path]:
    """Mapea protocol_id -> ruta de su package.yaml.

    Estructura esperada: protocols/<protocol_id>/<version>/package.yaml
    Si hay mas de una carpeta de version para el mismo protocolo, se usa la
    que ordena ultimo alfabeticamente (p.ej. v09 > v08).
    """
    found: dict[str, Path] = {}
    if not PROTOCOLS_DIR.is_dir():
        return found
    for protocol_dir in sorted(PROTOCOLS_DIR.iterdir()):
        if not protocol_dir.is_dir():
            continue
        version_dirs = sorted(p for p in protocol_dir.iterdir() if p.is_dir())
        for version_dir in reversed(version_dirs):
            candidate = version_dir / "package.yaml"
            if candidate.is_file():
                found[protocol_dir.name] = candidate
                break
    return found


def _load_uncached(protocol_id: str) -> ProtocolVersion:
    files = _discover_package_files()
    path = files.get(protocol_id)
    if path is None:
        raise KeyError(
            f"No existe package.yaml para el protocolo {protocol_id!r} en {PROTOCOLS_DIR}"
        )
    text = path.read_text(encoding="utf-8")
    package = load_package(text)
    logger.info("Registro de protocolos: cargado %s desde %s", protocol_id, path)
    return package


def get_package(protocol_id: str) -> ProtocolVersion:
    """Devuelve el ProtocolVersion del protocolo, cacheado tras la 1ra carga."""
    if protocol_id in _cache:
        return _cache[protocol_id]
    with _lock:
        if protocol_id not in _cache:
            _cache[protocol_id] = _load_uncached(protocol_id)
    return _cache[protocol_id]


def available_protocol_ids() -> list[str]:
    """IDs de todos los protocolos vendorizados (para el prompt del LLM)."""
    return sorted(_discover_package_files().keys())


def describe_available_protocols() -> list[Protocol]:
    """Metadatos (id/nombre/descripcion) de cada protocolo disponible, usados
    para construir dinamicamente el prompt del clasificador en
    tree_agent.select_protocol()."""
    return [get_package(pid).protocol for pid in available_protocol_ids()]
