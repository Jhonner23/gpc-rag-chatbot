"""Pruebas del segundo arbol del bosque nac-2026: criterios IDSA/ATS 2007 para
ingreso a UCI (Tabla 5, pag. 199).

Mismo objetivo que test_engine.py: confirmar que el JSON es un arbol valido
segun trees/schema.py y que el recorrido es 100% determinista -- sin ningun
LLM involucrado. Este arbol es el segundo caso real que valida el diseño de
"un arbol por protocolo/instrumento clinico" (CURB-65 fue el primero): ambos
viven en la misma guia (NAC 2026) pero son instrumentos de decision distintos
con logica de conteo distinta (5 preguntas aditivas vs. 2 mayores + 9 menores
con umbral de 3).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpc_rag.trees.engine import TreeSession, TreeValidationError, load_tree
from gpc_rag.trees.schema import DecisionNode, LeafNode

_TREE_PATH = (
    Path(__file__).parents[2]
    / "src"
    / "gpc_rag"
    / "trees"
    / "data"
    / "nac-2026"
    / "criterios-idsa-ats-uci.json"
)


def test_load_tree_valida_el_json():
    tree = load_tree(_TREE_PATH)

    assert tree.tree_id == "nac-2026-criterios-idsa-ats-uci"
    assert isinstance(tree.nodes[tree.root_node_id], DecisionNode)
    # 2 mayores + (9 menores x 3 variantes de conteo) + 3 hojas = 32 nodos
    assert len(tree.nodes) == 32


def test_cualquier_criterio_mayor_indica_uci_directa():
    tree = load_tree(_TREE_PATH)

    session = TreeSession(tree)
    session.answer(True)  # ventilacion mecanica invasiva
    hoja = session.current_node()
    assert isinstance(hoja, LeafNode)
    assert hoja.node_id == "leaf_uci_directa"

    session = TreeSession(tree)
    session.answer(False)  # no requiere VM
    session.answer(True)  # choque septico con vasopresores
    hoja = session.current_node()
    assert isinstance(hoja, LeafNode)
    assert hoja.node_id == "leaf_uci_directa"


def test_tres_criterios_menores_consecutivos_sugieren_uci():
    tree = load_tree(_TREE_PATH)
    session = TreeSession(tree)

    session.answer(False)  # major_vm_invasiva
    session.answer(False)  # major_choque_septico
    session.answer(True)  # minor 1
    session.answer(True)  # minor 2
    session.answer(True)  # minor 3 -> se alcanza el umbral, corta el arbol

    assert session.is_finished()
    hoja = session.current_node()
    assert isinstance(hoja, LeafNode)
    assert hoja.node_id == "leaf_uci_criterios_menores"
    # se corto antes de preguntar los 9 criterios menores
    assert len(session.trail) < 2 + 9 + 1


def test_tres_criterios_menores_dispersos_tambien_sugieren_uci():
    tree = load_tree(_TREE_PATH)
    session = TreeSession(tree)

    session.answer(False)
    session.answer(False)
    respuestas = [True, False, False, True, False, False, True, False, False]
    for r in respuestas:
        if session.is_finished():
            break
        session.answer(r)

    hoja = session.current_node()
    assert isinstance(hoja, LeafNode)
    assert hoja.node_id == "leaf_uci_criterios_menores"


def test_menos_de_tres_criterios_menores_no_indica_uci():
    tree = load_tree(_TREE_PATH)
    session = TreeSession(tree)

    session.answer(False)
    session.answer(False)
    for _ in range(9):
        session.answer(False)

    hoja = session.current_node()
    assert isinstance(hoja, LeafNode)
    assert hoja.node_id == "leaf_no_uci_por_criterios"


def test_dos_criterios_menores_no_alcanza_el_umbral():
    tree = load_tree(_TREE_PATH)
    session = TreeSession(tree)

    session.answer(False)
    session.answer(False)
    respuestas = [True, False, False, False, False, False, False, False, True]
    for r in respuestas:
        session.answer(r)

    hoja = session.current_node()
    assert isinstance(hoja, LeafNode)
    assert hoja.node_id == "leaf_no_uci_por_criterios"


def test_respuesta_invalida_lanza_error():
    tree = load_tree(_TREE_PATH)
    session = TreeSession(tree)

    with pytest.raises(TreeValidationError):
        session.answer("no_se")


def test_todos_los_nodos_hoja_tienen_cita_de_la_guia():
    tree = load_tree(_TREE_PATH)

    for node in tree.nodes.values():
        if isinstance(node, LeafNode):
            assert node.source.guide_file
            assert node.source_quote
