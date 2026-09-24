"""Logica del modo wizard, separada a proposito de Chainlit.

`chat/wizard.py` (que si importa chainlit) es una capa delgada que traduce lo
que devuelven estas funciones a mensajes y botones reales. Todo lo que decide
"que preguntar ahora" y "como se ve la hoja final" vive aqui, sin ninguna
dependencia de chainlit -- por eso se puede probar con pytest normal, sin
necesitar el framework de chat instalado (ver tests/trees/test_wizard_logic.py).

Nada en este modulo llama a un LLM: solo lee el estado de un TreeSession
(motor 100% determinista, trees/engine.py) y lo describe.
"""

from __future__ import annotations

from dataclasses import dataclass

from gpc_rag.trees.engine import TreeSession
from gpc_rag.trees.registry import TreeEntry
from gpc_rag.trees.schema import DecisionNode, LeafNode, Node


@dataclass(frozen=True)
class ChoiceOption:
    """Una opcion presentable al usuario (boton) para un nodo tipo enum."""

    value: str
    label: str


@dataclass(frozen=True)
class QuestionStep:
    """Lo que hay que preguntarle al usuario para el nodo actual del arbol."""

    node_id: str
    question: str
    input_type: str  # "boolean" | "numeric" | "enum"
    options: list[ChoiceOption] | None = None  # solo para enum


@dataclass(frozen=True)
class LeafReached:
    """El arbol llego a un nodo hoja: hay una recomendacion final que mostrar."""

    node_id: str
    recommendation: str
    evidence_level: str | None
    citation_markdown: str


@dataclass(frozen=True)
class TreeMenuOption:
    """Una entrada del menu inicial del wizard (que arbol/protocolo evaluar)."""

    tree_id: str
    title: str
    description: str


def build_tree_menu(entries: list[TreeEntry]) -> list[TreeMenuOption]:
    return [
        TreeMenuOption(tree_id=e.tree.tree_id, title=e.tree.title, description=e.tree.description)
        for e in entries
    ]


def format_citation(guide_file: str, section: str, page: int | None, quote: str) -> str:
    pagina = f", pág. {page}" if page is not None else ""
    return f"**Fuente:** {guide_file} — {section}{pagina}\n\n> {quote}"


def describe_current_node(session: TreeSession) -> QuestionStep | LeafReached:
    """Traduce el nodo actual de la sesion a algo que la UI pueda renderizar."""
    node: Node = session.current_node()

    if isinstance(node, LeafNode):
        return LeafReached(
            node_id=node.node_id,
            recommendation=node.recommendation,
            evidence_level=node.evidence_level,
            citation_markdown=format_citation(
                node.source.guide_file, node.source.section, node.source.page, node.source_quote
            ),
        )

    assert isinstance(node, DecisionNode)
    options: list[ChoiceOption] | None = None
    if node.input_type == "enum":
        options = [ChoiceOption(value=str(o.value), label=o.label) for o in (node.options or [])]

    return QuestionStep(
        node_id=node.node_id,
        question=node.question,
        input_type=node.input_type,
        options=options,
    )


_TRUE_VALUES = {"si", "sí", "true", "yes"}
_FALSE_VALUES = {"no", "false"}


def parse_boolean_answer(raw: str) -> bool:
    """Convierte el valor de un boton Si/No a bool. Lanza ValueError si no matchea."""
    normalizado = raw.strip().lower()
    if normalizado in _TRUE_VALUES:
        return True
    if normalizado in _FALSE_VALUES:
        return False
    raise ValueError(f"valor de respuesta booleana no reconocido: {raw!r}")


def parse_numeric_answer(raw: str) -> float:
    """Convierte texto libre del usuario a numero. Lanza ValueError si no es valido."""
    normalizado = raw.strip().replace(",", ".")
    try:
        return float(normalizado)
    except ValueError as exc:
        raise ValueError(f"'{raw}' no es un numero valido") from exc
