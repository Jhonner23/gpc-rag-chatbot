"""Respuesta del RAG por tipo de fuente (generation/by_source.py)."""

from __future__ import annotations

from gpc_rag.common.models import Chunk, RetrievedChunk
from gpc_rag.generation.by_source import compose, generate_by_source

_REFUSAL = "No encuentro informacion suficiente en las guias disponibles para responder esto con seguridad."


def _rc(i: int, kind: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(text=f"t{i}", source_file=f"{kind}{i}.pdf", chunk_id=str(i), source_type=kind),
        score=1.0,
    )


class _FakeGenerator:
    def __init__(self, replies: dict[str, str]):
        self.replies = replies
        self.calls: list[tuple[set[str], str | None]] = []

    def generate(self, question, chunks, extra_instruction=None):
        kinds = {rc.chunk.source_type for rc in chunks}
        self.calls.append((kinds, extra_instruction))
        key = "institucional" if kinds == {"institucional"} else "externas" if "institucional" not in kinds else "todo"
        return self.replies[key]


def test_each_pass_sees_only_its_own_sources_and_the_answer_is_one_text() -> None:
    gen = _FakeGenerator({"institucional": "Tratar solo en gestantes.", "externas": "IDSA: no tratar."})
    sources = [_rc(1, "institucional"), _rc(2, "guia"), _rc(3, "revision")]
    text = generate_by_source(gen, "¿Cuándo tratar?", sources, extra_instruction="Sé estricto.")
    assert [kinds for kinds, _ in gen.calls] == [{"institucional"}, {"guia", "revision"}]
    assert all("Sé estricto." in extra for _, extra in gen.calls)
    assert text == "Tratar solo en gestantes.\n\nIDSA: no tratar."  # una sola respuesta, sin encabezados


def test_only_institutional_sources_keep_the_single_pass() -> None:
    gen = _FakeGenerator({"institucional": "Tratar solo en gestantes.", "todo": "x"})
    text = generate_by_source(gen, "q", [_rc(1, "institucional")])
    assert text == "Tratar solo en gestantes."
    assert gen.calls == [({"institucional"}, None)]


def test_refusals_are_replaced_or_kept_as_abstention() -> None:
    assert compose(_REFUSAL, "Amoxicilina [Guia: ats.pdf].") == "Amoxicilina [Guia: ats.pdf]."
    assert compose(_REFUSAL, _REFUSAL) == _REFUSAL
    assert compose("Protocolo: X.", _REFUSAL) == "Protocolo: X."
