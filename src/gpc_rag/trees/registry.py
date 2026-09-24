"""Registro del bosque de arboles disponibles.

Descubre todos los arboles JSON bajo trees/data/**/*.json, los valida contra
el schema y los deja listos para que el modo wizard del chat los ofrezca al
usuario (ver chat/wizard.py). Cargar/validar un arbol es barato (JSON +
Pydantic), asi que se hace una vez por proceso, no en cada turno de
conversacion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gpc_rag.trees.engine import load_tree
from gpc_rag.trees.schema import DecisionTree

_DATA_DIR = Path(__file__).parent / "data"


@dataclass(frozen=True)
class TreeEntry:
    """Un arbol del bosque, ya cargado y validado."""

    tree_id: str
    tree: DecisionTree
    path: Path


def _discover(data_dir: Path) -> dict[str, TreeEntry]:
    entries: dict[str, TreeEntry] = {}
    for json_path in sorted(data_dir.rglob("*.json")):
        tree = load_tree(json_path)
        if tree.tree_id in entries:
            raise ValueError(
                f"tree_id duplicado {tree.tree_id!r}: {entries[tree.tree_id].path} y {json_path}"
            )
        entries[tree.tree_id] = TreeEntry(tree_id=tree.tree_id, tree=tree, path=json_path)
    return entries


_registry: dict[str, TreeEntry] | None = None


def get_registry(data_dir: Path | None = None, *, force_reload: bool = False) -> dict[str, TreeEntry]:
    """Devuelve el registro (tree_id -> TreeEntry), cacheado tras la primera llamada.

    `data_dir` y `force_reload` existen sobre todo para pruebas (poder apuntar
    a un directorio temporal con arboles de ejemplo sin tocar el cache global).
    """
    global _registry
    if _registry is None or force_reload or data_dir is not None:
        loaded = _discover(data_dir or _DATA_DIR)
        if data_dir is None:
            _registry = loaded
        return loaded
    return _registry


def list_trees(data_dir: Path | None = None) -> list[TreeEntry]:
    return list(get_registry(data_dir).values())


def get_tree(tree_id: str, data_dir: Path | None = None) -> DecisionTree:
    return get_registry(data_dir)[tree_id].tree
