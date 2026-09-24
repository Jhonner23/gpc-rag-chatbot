"""Pruebas del esquema (schema.py) y del motor de recorrido (engine.py).

Usan el arbol de ejemplo `trees/data/nac-2026/severidad-hospitalizacion.json`
(CURB-65) para verificar que el JSON es una forma valida de arbol y que la
navegacion es 100% determinista -- sin ningun mock de LLM, porque el motor
no llama a ninguno.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from gpc_rag.trees.engine import TreeSession, TreeValidationError, load_tree
from gpc_rag.trees.schema import Branch, DecisionNode, DecisionTree, LeafNode

_EXAMPLE_TREE_PATH = (
    Path(__file__).parents[2] / "src" / "gpc_rag" / "trees" / "data" / "nac-2026" / "severidad-hospitalizacion.json"
)


def test_load_tree_valida_el_json_de_ejemplo():
    tree = load_tree(_EXAMPLE_TREE_PATH)

    assert tree.tree_id == "nac-2026-severidad-hospitalizacion"
    assert isinstance(tree.nodes[tree.root_node_id], DecisionNode)


def test_recorrido_completo_todo_no_llega_a_riesgo_bajo():
    tree = load_tree(_EXAMPLE_TREE_PATH)
    session = TreeSession(tree)

    for _ in range(5):  # confusion, urea, FR, PA, edad -- las 5 preguntas CURB-65
        assert not session.is_finished()
        session.answer(False)

    assert session.is_finished()
    hoja = session.current_node()
    assert isinstance(hoja, LeafNode)
    assert hoja.node_id == "leaf_score0"
    assert "manejo ambulatorio" in hoja.recommendation.lower()


def test_recorrido_completo_todo_si_llega_a_riesgo_alto():
    tree = load_tree(_EXAMPLE_TREE_PATH)
    session = TreeSession(tree)

    for _ in range(5):
        session.answer(True)

    hoja = session.current_node()
    assert isinstance(hoja, LeafNode)
    assert hoja.node_id == "leaf_score5"
    assert "uci" in hoja.recommendation.lower()


def test_puntaje_dos_llega_a_riesgo_intermedio():
    tree = load_tree(_EXAMPLE_TREE_PATH)
    session = TreeSession(tree)

    # confusion=True, urea=True, FR=False, PA=False, edad=False -> CURB-65 = 2
    session.answer(True)
    session.answer(True)
    session.answer(False)
    session.answer(False)
    session.answer(False)

    hoja = session.current_node()
    assert isinstance(hoja, LeafNode)
    assert hoja.node_id == "leaf_score2"
    assert "considerar hospitalizacion" in hoja.recommendation.lower()


def test_respuesta_que_no_matchea_ningun_branch_lanza_error():
    tree = load_tree(_EXAMPLE_TREE_PATH)
    session = TreeSession(tree)

    with pytest.raises(TreeValidationError):
        session.answer("tal_vez")  # los branches solo aceptan True/False


def test_responder_una_hoja_lanza_error():
    tree = load_tree(_EXAMPLE_TREE_PATH)
    session = TreeSession(tree)
    for _ in range(5):
        session.answer(False)

    with pytest.raises(TreeValidationError):
        session.answer(True)


def test_reset_vuelve_a_la_raiz():
    tree = load_tree(_EXAMPLE_TREE_PATH)
    session = TreeSession(tree)
    session.answer(True)
    assert session.current_node_id != tree.root_node_id

    session.reset()
    assert session.current_node_id == tree.root_node_id
    assert session.answers == {}
    assert session.trail == [tree.root_node_id]


def test_arbol_con_branch_a_nodo_inexistente_no_valida():
    with pytest.raises(ValidationError):
        DecisionTree.model_validate(
            {
                "tree_id": "t",
                "title": "t",
                "description": "d",
                "gpc_source": "g.pdf",
                "root_node_id": "a",
                "nodes": {
                    "a": {
                        "type": "decision",
                        "node_id": "a",
                        "question": "?",
                        "variable": "x",
                        "input_type": "boolean",
                        "branches": [
                            {"operator": "eq", "value": True, "next_node_id": "no_existe"},
                        ],
                    },
                },
            }
        )


def test_nodo_enum_sin_options_no_valida():
    with pytest.raises(ValidationError):
        DecisionNode.model_validate(
            {
                "node_id": "a",
                "question": "?",
                "variable": "x",
                "input_type": "enum",
                "branches": [{"operator": "eq", "value": "x", "next_node_id": "b"}],
            }
        )


def test_branch_operador_between():
    branch = Branch(operator="between", value=[2, 4], next_node_id="siguiente")
    tree = DecisionTree.model_validate(
        {
            "tree_id": "t",
            "title": "t",
            "description": "d",
            "gpc_source": "g.pdf",
            "root_node_id": "raiz",
            "nodes": {
                "raiz": {
                    "type": "decision",
                    "node_id": "raiz",
                    "question": "puntaje?",
                    "variable": "score",
                    "input_type": "numeric",
                    "branches": [branch.model_dump()],
                },
                "siguiente": {
                    "type": "leaf",
                    "node_id": "siguiente",
                    "recommendation": "ok",
                    "source": {"guide_file": "g.pdf", "section": "s", "page": 1},
                    "source_quote": "q",
                },
            },
        }
    )
    session = TreeSession(tree)
    hoja = session.answer(3)
    assert isinstance(hoja, LeafNode)
    assert hoja.node_id == "siguiente"
