"""Pruebas de trees/wizard_logic.py y trees/registry.py.

Deliberadamente no importan chainlit -- esta capa es pura Python sobre
TreeSession/DecisionTree, por eso se puede probar sin instalar el framework
de chat. chat/wizard.py (la capa delgada que si usa chainlit) solo traduce
lo que estas funciones devuelven a mensajes/botones reales.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpc_rag.trees.engine import TreeSession, load_tree
from gpc_rag.trees.registry import get_registry, list_trees
from gpc_rag.trees.wizard_logic import (
    LeafReached,
    QuestionStep,
    build_tree_menu,
    describe_current_node,
    parse_boolean_answer,
    parse_numeric_answer,
)

_DATA_DIR = Path(__file__).parents[2] / "src" / "gpc_rag" / "trees" / "data"
_CURB65_PATH = _DATA_DIR / "nac-2026" / "severidad-hospitalizacion.json"


def test_registro_descubre_los_dos_arboles_del_bosque_nac_2026():
    entries = list_trees()
    tree_ids = {e.tree_id for e in entries}

    assert "nac-2026-severidad-hospitalizacion" in tree_ids
    assert "nac-2026-criterios-idsa-ats-uci" in tree_ids


def test_registro_es_un_singleton_cacheado():
    primero = get_registry()
    segundo = get_registry()
    assert primero is segundo  # mismo objeto: no relee el disco en cada llamada


def test_menu_expone_titulo_y_descripcion_de_cada_arbol():
    menu = build_tree_menu(list_trees())
    opciones_por_id = {o.tree_id: o for o in menu}

    curb65 = opciones_por_id["nac-2026-severidad-hospitalizacion"]
    assert "CURB-65" in curb65.title
    assert curb65.description  # no vacio


def test_describe_current_node_nodo_de_decision_booleana():
    tree = load_tree(_CURB65_PATH)
    session = TreeSession(tree)

    paso = describe_current_node(session)

    assert isinstance(paso, QuestionStep)
    assert paso.input_type == "boolean"
    assert paso.node_id == tree.root_node_id
    assert paso.question  # tiene texto de pregunta


def test_describe_current_node_hoja_incluye_cita_formateada():
    tree = load_tree(_CURB65_PATH)
    session = TreeSession(tree)
    for _ in range(5):
        session.answer(False)  # CURB-65 = 0 -> leaf_score0

    resultado = describe_current_node(session)

    assert isinstance(resultado, LeafReached)
    assert resultado.node_id == "leaf_score0"
    assert "manejo ambulatorio" in resultado.recommendation.lower()
    assert "Fuente:" in resultado.citation_markdown
    assert "pág." in resultado.citation_markdown


@pytest.mark.parametrize("raw,esperado", [("Si", True), ("sí", True), ("SI", True), ("no", False), ("No", False)])
def test_parse_boolean_answer_reconoce_variantes(raw, esperado):
    assert parse_boolean_answer(raw) is esperado


def test_parse_boolean_answer_valor_no_reconocido_lanza_error():
    with pytest.raises(ValueError):
        parse_boolean_answer("tal vez")


def test_parse_numeric_answer_acepta_coma_decimal():
    assert parse_numeric_answer("36,5") == pytest.approx(36.5)


def test_parse_numeric_answer_texto_invalido_lanza_error():
    with pytest.raises(ValueError):
        parse_numeric_answer("no es un numero")
