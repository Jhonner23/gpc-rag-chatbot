"""Modo wizard del chat: evaluacion guiada con un arbol de decision.

Capa delgada sobre `trees/wizard_logic.py` -- aqui solo se traduce lo que esa
capa (pura, sin chainlit) devuelve a mensajes y botones reales de Chainlit.
Ningun LLM participa en este modo: la navegacion del arbol la hace
`TreeSession` (trees/engine.py), 100% determinista.

Se activa desde chat/app.py cuando el usuario elige el chat profile
"Evaluacion guiada" (ver sección 10 de la arquitectura, Opcion 1: modos
separados -- chat libre = RAG, evaluacion guiada = arbol).
"""

from __future__ import annotations

import chainlit as cl

from gpc_rag.trees.engine import TreeSession, TreeValidationError
from gpc_rag.trees.registry import list_trees
from gpc_rag.trees.wizard_logic import (
    LeafReached,
    QuestionStep,
    build_tree_menu,
    describe_current_node,
    parse_boolean_answer,
    parse_numeric_answer,
)

_SESSION_KEY = "tree_session"
_AWAITING_NUMERIC_KEY = "tree_awaiting_numeric_node_id"


async def start() -> None:
    """Punto de entrada del modo wizard: se llama desde on_chat_start."""
    cl.user_session.set(_SESSION_KEY, None)
    cl.user_session.set(_AWAITING_NUMERIC_KEY, None)
    await cl.Message(
        content=(
            "Modo **evaluacion guiada**: te hago preguntas puntuales, una a la vez, "
            "y al final te doy una recomendacion citando la guia exacta -- sin usar "
            "ningun modelo de lenguaje en el camino, solo la logica del arbol de "
            "decision. ¿Que protocolo quieres evaluar?"
        )
    ).send()
    await _show_tree_menu()


async def _show_tree_menu() -> None:
    entries = list_trees()
    opciones = build_tree_menu(entries)

    actions = [
        cl.Action(
            name="tree_select_root",
            payload={"tree_id": opcion.tree_id},
            label=opcion.title,
        )
        for opcion in opciones
    ]
    descripciones = "\n\n".join(f"**{o.title}**\n{o.description}" for o in opciones)
    await cl.Message(content=descripciones, actions=actions).send()


@cl.action_callback("tree_select_root")
async def on_select_tree(action: cl.Action) -> None:
    tree_id = action.payload["tree_id"]
    entries = {e.tree_id: e for e in list_trees()}
    entry = entries[tree_id]

    session = TreeSession(entry.tree)
    cl.user_session.set(_SESSION_KEY, session)
    cl.user_session.set(_AWAITING_NUMERIC_KEY, None)

    await cl.Message(content=f"Iniciando: **{entry.tree.title}**").send()
    await _ask_current_step()


async def _ask_current_step() -> None:
    session: TreeSession | None = cl.user_session.get(_SESSION_KEY)
    if session is None:
        await _show_tree_menu()
        return

    paso = describe_current_node(session)

    if isinstance(paso, LeafReached):
        await _render_leaf(paso)
        return

    assert isinstance(paso, QuestionStep)
    if paso.input_type == "boolean":
        actions = [
            cl.Action(name="tree_bool_answer", payload={"value": True}, label="Sí"),
            cl.Action(name="tree_bool_answer", payload={"value": False}, label="No"),
        ]
        cl.user_session.set(_AWAITING_NUMERIC_KEY, None)
        await cl.Message(content=paso.question, actions=actions).send()
    elif paso.input_type == "enum":
        actions = [
            cl.Action(name="tree_enum_answer", payload={"value": o.value}, label=o.label)
            for o in (paso.options or [])
        ]
        cl.user_session.set(_AWAITING_NUMERIC_KEY, None)
        await cl.Message(content=paso.question, actions=actions).send()
    else:  # numeric: no hay boton posible, se pide como texto libre
        cl.user_session.set(_AWAITING_NUMERIC_KEY, paso.node_id)
        await cl.Message(content=f"{paso.question}\n\n_(responde con un numero)_").send()


async def _render_leaf(leaf: LeafReached) -> None:
    nivel = f"\n\n**Nivel de evidencia:** {leaf.evidence_level}" if leaf.evidence_level else ""
    contenido = (
        f"### Recomendación\n{leaf.recommendation}{nivel}\n\n{leaf.citation_markdown}\n\n"
        "_Esta evaluación apoya pero no reemplaza el juicio clínico._"
    )
    cl.user_session.set(_AWAITING_NUMERIC_KEY, None)
    actions = [cl.Action(name="tree_restart", payload={}, label="Evaluar otro protocolo")]
    await cl.Message(content=contenido, actions=actions).send()


@cl.action_callback("tree_bool_answer")
async def on_bool_answer(action: cl.Action) -> None:
    await _apply_answer(bool(action.payload["value"]))


@cl.action_callback("tree_enum_answer")
async def on_enum_answer(action: cl.Action) -> None:
    await _apply_answer(action.payload["value"])


@cl.action_callback("tree_restart")
async def on_restart(_action: cl.Action) -> None:
    cl.user_session.set(_SESSION_KEY, None)
    cl.user_session.set(_AWAITING_NUMERIC_KEY, None)
    await _show_tree_menu()


async def _apply_answer(value: bool | float | str) -> None:
    session: TreeSession | None = cl.user_session.get(_SESSION_KEY)
    if session is None:
        await _show_tree_menu()
        return

    try:
        session.answer(value)
    except TreeValidationError as exc:
        await cl.Message(content=f"No pude registrar esa respuesta ({exc}). Intenta de nuevo.").send()
        await _ask_current_step()
        return

    await _ask_current_step()


async def handle_text_message(message: cl.Message) -> bool:
    """Procesa un mensaje de texto libre si el wizard esta esperando un numero.

    Devuelve True si el mensaje era para el wizard (ya se manejo), False si no
    aplica y el caller (chat/app.py) debe tratarlo como una pregunta normal.
    """
    awaiting_node_id = cl.user_session.get(_AWAITING_NUMERIC_KEY)
    if not awaiting_node_id:
        return False

    session: TreeSession | None = cl.user_session.get(_SESSION_KEY)
    if session is None:
        return False

    try:
        valor = parse_numeric_answer(message.content)
    except ValueError as exc:
        await cl.Message(content=f"{exc}. Intenta escribir solo el numero.").send()
        return True

    await _apply_answer(valor)
    return True
