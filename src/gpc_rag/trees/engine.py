"""Motor de ejecucion del arbol de decision -- 100% determinista, sin LLM.

Es el componente que cumple el requisito explicito de la asesora: el arbol
debe poder navegarse/decidirse sin invocar un LLM en el camino critico. Todo
lo que hace esta clase es evaluar los `branches` del nodo actual contra la
respuesta del usuario y saltar al `next_node_id` correspondiente -- logica
Python pura sobre el JSON validado por `trees/schema.py`.

El LLM solo participa offline, al construir el arbol (scripts/extract_tree.py).
"""

from __future__ import annotations

import json
from pathlib import Path

from gpc_rag.trees.schema import Branch, DecisionNode, DecisionTree, LeafNode, Node


class TreeValidationError(ValueError):
    """La respuesta del usuario no matchea ningun branch del nodo actual."""


def load_tree(path: str | Path) -> DecisionTree:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return DecisionTree.model_validate(data)


def _branch_matches(branch: Branch, answer: bool | int | float | str) -> bool:
    op = branch.operator
    value = branch.value

    if op == "eq":
        return answer == value
    if op == "ne":
        return answer != value
    if op == "in":
        assert isinstance(value, list)
        return answer in value
    if op == "between":
        assert isinstance(value, list) and len(value) == 2
        low, high = value
        return isinstance(answer, (int, float)) and low <= answer <= high  # type: ignore[operator]

    # comparaciones numericas: gt/gte/lt/lte
    if not isinstance(answer, (int, float)) or not isinstance(value, (int, float)):
        return False
    if op == "gt":
        return answer > value
    if op == "gte":
        return answer >= value
    if op == "lt":
        return answer < value
    if op == "lte":
        return answer <= value
    raise ValueError(f"operador desconocido: {op}")  # pragma: no cover -- Literal ya lo restringe


class TreeSession:
    """Sesion de recorrido de un arbol para una conversacion puntual.

    Uso tipico (agente wizard, ver agents/tree_agent.py):
        session = TreeSession(tree)
        node = session.current_node()          # -> DecisionNode, hacer la pregunta
        node = session.answer(respuesta_usuario)  # -> DecisionNode o LeafNode
        ...hasta que isinstance(node, LeafNode)   # -> mostrar recomendacion + cita
    """

    def __init__(self, tree: DecisionTree):
        self.tree = tree
        self.current_node_id = tree.root_node_id
        self.trail: list[str] = [tree.root_node_id]
        self.answers: dict[str, bool | int | float | str] = {}

    def current_node(self) -> Node:
        return self.tree.nodes[self.current_node_id]

    def is_finished(self) -> bool:
        return isinstance(self.current_node(), LeafNode)

    def answer(self, value: bool | int | float | str) -> Node:
        node = self.current_node()
        if not isinstance(node, DecisionNode):
            raise TreeValidationError(f"el nodo {node.node_id!r} ya es una hoja, no acepta respuestas")

        for branch in node.branches:
            if _branch_matches(branch, value):
                self.answers[node.variable] = value
                self.current_node_id = branch.next_node_id
                self.trail.append(self.current_node_id)
                return self.current_node()

        raise TreeValidationError(
            f"la respuesta {value!r} no matchea ningun branch del nodo {node.node_id!r} "
            f"(variable={node.variable!r})"
        )

    def reset(self) -> None:
        self.current_node_id = self.tree.root_node_id
        self.trail = [self.tree.root_node_id]
        self.answers = {}
