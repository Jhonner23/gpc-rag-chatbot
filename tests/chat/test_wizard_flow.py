"""Prueba el flujo del modo wizard end-to-end (chat/wizard.py) usando el
chainlit de mentira de fake_chainlit.py.

A diferencia de tests/trees/test_wizard_logic.py (que prueba la logica pura,
sin chainlit para nada), esto ejercita chat/wizard.py tal cual -- la capa que
SI importa chainlit -- para confirmar que el cableado de acciones/mensajes es
correcto: que boton llama a que callback, que payload se manda, que texto
final se muestra. Sigue sin haber ningun LLM en el camino: se prueba que el
wizard llega a la hoja correcta con las mismas respuestas que ya se probaron
en tests/trees/test_idsa_ats_tree.py y test_engine.py.
"""

from __future__ import annotations

import asyncio

import pytest

import fake_chainlit as cl  # instalado como sys.modules["chainlit"] por conftest.py

from gpc_rag.chat import wizard
from gpc_rag.trees.engine import TreeSession
from gpc_rag.trees.schema import DecisionTree


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _reset_fake_chainlit():
    cl.reset()
    yield
    cl.reset()


def test_start_muestra_intro_y_menu_con_los_dos_arboles():
    run(wizard.start())

    contenidos = [m.content for m in cl.sent_messages]
    assert any("evaluacion guiada" in c.lower() for c in contenidos)

    mensaje_menu = cl.sent_messages[-1]
    etiquetas = {a.label for a in mensaje_menu.actions}
    assert any("CURB-65" in e for e in etiquetas)
    assert any("IDSA/ATS" in e for e in etiquetas)


def test_seleccionar_curb65_hace_la_primera_pregunta_booleana():
    run(wizard.start())

    action = cl.Action(name="tree_select_root", payload={"tree_id": "nac-2026-severidad-hospitalizacion"})
    run(wizard.on_select_tree(action))

    ultima = cl.sent_messages[-1]
    assert "confusion" in ultima.content.lower()
    etiquetas = {a.label for a in ultima.actions}
    assert etiquetas == {"Sí", "No"}
    assert {a.name for a in ultima.actions} == {"tree_bool_answer"}


def test_recorrido_completo_con_botones_llega_a_la_hoja_correcta():
    run(wizard.start())
    run(
        wizard.on_select_tree(
            cl.Action(name="tree_select_root", payload={"tree_id": "nac-2026-severidad-hospitalizacion"})
        )
    )

    # Responder "No" (False) a las 5 preguntas via el mismo callback que
    # dispara un boton real -> debe llegar a leaf_score0 (riesgo bajo).
    for _ in range(5):
        run(wizard.on_bool_answer(cl.Action(name="tree_bool_answer", payload={"value": False})))

    hoja = cl.sent_messages[-1]
    assert "Recomendación" in hoja.content
    assert "manejo ambulatorio" in hoja.content.lower()
    assert "Fuente:" in hoja.content
    assert any(a.name == "tree_restart" for a in hoja.actions)


def test_reiniciar_vuelve_al_menu():
    run(wizard.start())
    run(
        wizard.on_select_tree(
            cl.Action(name="tree_select_root", payload={"tree_id": "nac-2026-severidad-hospitalizacion"})
        )
    )
    for _ in range(5):
        run(wizard.on_bool_answer(cl.Action(name="tree_bool_answer", payload={"value": False})))

    run(wizard.on_restart(cl.Action(name="tree_restart", payload={})))

    ultima = cl.sent_messages[-1]
    etiquetas = {a.label for a in ultima.actions}
    assert any("CURB-65" in e for e in etiquetas)


def test_handle_text_message_no_aplica_si_no_hay_pregunta_numerica_pendiente():
    run(wizard.start())
    mensaje = cl.Message(content="hola")

    aplico = run(wizard.handle_text_message(mensaje))

    assert aplico is False


def test_pregunta_numerica_se_responde_por_texto_libre():
    # Arbol sintetico minimo con un nodo numerico, para probar la rama que
    # ningun arbol real usa todavia (ambos arboles actuales son booleanos).
    tree = DecisionTree.model_validate(
        {
            "tree_id": "t-numerico",
            "title": "t",
            "description": "d",
            "gpc_source": "g.pdf",
            "root_node_id": "raiz",
            "nodes": {
                "raiz": {
                    "type": "decision",
                    "node_id": "raiz",
                    "question": "Cual es la temperatura del paciente?",
                    "variable": "temperatura",
                    "input_type": "numeric",
                    "branches": [
                        {"operator": "gte", "value": 38.0, "next_node_id": "hoja"},
                        {"operator": "lt", "value": 38.0, "next_node_id": "hoja"},
                    ],
                },
                "hoja": {
                    "type": "leaf",
                    "node_id": "hoja",
                    "recommendation": "ok",
                    "source": {"guide_file": "g.pdf", "section": "s", "page": 1},
                    "source_quote": "q",
                },
            },
        }
    )
    cl.user_session.set("tree_session", TreeSession(tree))
    run(wizard._ask_current_step())

    pregunta = cl.sent_messages[-1]
    assert "numero" in pregunta.content.lower()

    aplico = run(wizard.handle_text_message(cl.Message(content="38,5")))

    assert aplico is True
    hoja = cl.sent_messages[-1]
    assert "Recomendación" in hoja.content
