"""Reintento del cliente de Ollama (common/ollama_client.py)."""

from __future__ import annotations

import pytest
from ollama import ResponseError

from gpc_rag.common import ollama_client


def _client(monkeypatch, errors):
    calls = []

    def fake_chat(self, *args, **kwargs):
        calls.append(kwargs)
        if errors:
            raise errors.pop(0)
        return {"message": {"content": "ok"}}

    monkeypatch.setattr(ollama_client._OllamaClient, "chat", fake_chat)
    monkeypatch.setattr(ollama_client, "RETRY_DELAY_SECONDS", 0)
    return ollama_client.Client(host="http://localhost:11434"), calls


def test_server_error_is_retried_once(monkeypatch) -> None:
    client, calls = _client(monkeypatch, [ResponseError("connection refused", 500)])
    assert client.chat(model="m", messages=[])["message"]["content"] == "ok"
    assert len(calls) == 2


def test_client_error_is_not_retried(monkeypatch) -> None:
    client, calls = _client(monkeypatch, [ResponseError("bad request", 400)])
    with pytest.raises(ResponseError):
        client.chat(model="m", messages=[])
    assert len(calls) == 1
