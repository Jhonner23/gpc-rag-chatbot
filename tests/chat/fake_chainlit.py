"""Chainlit de mentira, solo para pruebas de chat/wizard.py y chat/app.py.

Implementa unicamente lo que ese codigo usa: Action, Message (con
send()/update() async), user_session (dict simple), action_callback/
on_chat_start/on_message/set_chat_profiles (decoradores que no hacen nada --
en las pruebas se llama a las funciones directamente, no via un servidor
Chainlit real) y ChatProfile. No es un reemplazo de Chainlit: es un doble de
prueba minimo para validar la LOGICA propia (que boton dispara que accion,
que texto se muestra, que payload se manda) sin necesitar el framework de
chat instalado ni un servidor corriendo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

sent_messages: list["Message"] = []


class _UserSession:
    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def clear(self) -> None:
        self._data.clear()


user_session = _UserSession()


@dataclass
class Action:
    name: str
    payload: dict
    label: str = ""


@dataclass
class ChatProfile:
    name: str
    markdown_description: str = ""
    icon: str | None = None


@dataclass
class Message:
    content: str
    actions: list[Action] = field(default_factory=list)

    async def send(self) -> Message:
        sent_messages.append(self)
        return self

    async def update(self) -> Message:
        sent_messages.append(self)
        return self


def action_callback(_name: str):
    def decorator(func):
        return func

    return decorator


def on_chat_start(func):
    return func


def on_message(func):
    return func


def set_chat_profiles(func):
    return func


def reset() -> None:
    """Limpia el estado global entre pruebas (mensajes enviados + sesion)."""
    sent_messages.clear()
    user_session.clear()
