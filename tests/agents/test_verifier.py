"""Pruebas del verificador de evidencia (sin LLM, deterministas)."""

from __future__ import annotations

from unittest.mock import patch

from omegaconf import OmegaConf

from gpc_rag.agents.verifier import verify_contributions
from gpc_rag.common.models import Chunk, RetrievedChunk

_CFG = OmegaConf.create(
    {
        "env": {"ollama_url": "http://localhost:11434"},
        "agents": {"evaluator_model": "fake"},
    }
)


def _chat(text: str) -> dict:
    return {"message": {"content": text}}


def _verify(state: dict, judge: str = '{"contradiccion": false}') -> dict:
    with patch("gpc_rag.agents.verifier.Client") as mock_client:
        mock_client.return_value.chat.return_value = _chat(judge)
        return verify_contributions(state, _CFG)


def _tree_result(source_file: str, page: int | None = 4) -> dict:
    return {
        "protocol_id": "CT-PL-193",
        "protocol_version": "9",
        "rule_id": "rule_x",
        "answer_text": "Segun el arbol de decision del protocolo CT-PL-193: ...",
        "citations": [
            {
                "source_file": source_file,
                "section": "S",
                "page": page,
                "verbatim_text": "texto fuente",
            }
        ],
    }


def _rag_sources(source_file: str) -> list[RetrievedChunk]:
    chunk = Chunk(text="t", source_file=source_file, section="S", page=4)
    return [RetrievedChunk(chunk=chunk, score=0.9, retrieval_method="rerank")]


def _state(tree_file: str, rag_file: str, page: int | None = 4) -> dict:
    return {
        "question": "q",
        "tree_protocol_id": "CT-PL-193",
        "tree_result": _tree_result(tree_file, page),
        "answer": "Respuesta del RAG.",
        "sources": _rag_sources(rag_file),
        "eval_ok": True,
        "eval_status": "aprobada",
    }


def test_both_contributions_eligible_without_discrepancies() -> None:
    out = _verify(
        _state(
            "CT-PL-193 PROTOCOLO NAC (v9-ago-2024).pdf",
            "CT-PL-193_Protocolo_NAC_v9-ago-2024.pdf",
        )
    )
    tree, rag = out["contributions"]
    assert (
        tree["status"] == "respuesta_fundamentada"
        and tree["verification"] == "verificada"
    )
    assert (
        rag["status"] == "respuesta_fundamentada"
        and rag["verification"] == "verificada"
    )
    assert tree["eligible"] and rag["eligible"]
    assert out["discrepancies"] == []


def test_version_discrepancy_is_reported_not_resolved() -> None:
    out = _verify(
        _state(
            "CT-PL-193 PROTOCOLO NAC (v9-ago-2024).pdf",
            "CT-PL-193_Protocolo_NAC_v8-jun-2020.pdf",
        )
    )
    assert len(out["discrepancies"]) == 1
    assert "CT-PL-193" in out["discrepancies"][0]
    assert all(c["eligible"] for c in out["contributions"])


def test_tree_without_complete_citation_is_not_eligible() -> None:
    out = _verify(_state("CT-PL-193 (v9).pdf", "CT-PL-193_v9.pdf", page=None))
    tree, _ = out["contributions"]
    assert tree["status"] == "respuesta_fundamentada"
    assert tree["verification"] == "sin_procedencia"
    assert not tree["eligible"]


def test_rag_abstention_is_not_eligible_content() -> None:
    state = _state("CT-PL-193 (v9).pdf", "CT-PL-193_v9.pdf")
    state["answer"] = "No encontré información suficiente en las guías."
    _, rag = _verify(state)["contributions"]
    assert rag["status"] == "abstencion"
    assert not rag["eligible"]


def test_content_contradiction_is_reported() -> None:
    out = _verify(
        _state("CT-PL-193 (v9).pdf", "CT-PL-193_v9.pdf"),
        judge='{"contradiccion": true, "detalle": "difieren en la duracion."}',
    )
    assert any(
        "Posible contradiccion" in d and "duracion" in d for d in out["discrepancies"]
    )


def test_unreadable_contradiction_verdict_is_reported_as_not_checked() -> None:
    out = _verify(_state("CT-PL-193 (v9).pdf", "CT-PL-193_v9.pdf"), judge="no es json")
    assert any("No fue posible comprobar" in d for d in out["discrepancies"])


def test_tree_outcome_states() -> None:
    state = _state("CT-PL-193 (v9).pdf", "CT-PL-193_v9.pdf")
    state["tree_result"] = None
    state["tree_outcome"] = "informacion_faltante"
    tree, _ = _verify(state)["contributions"]
    assert tree["status"] == "informacion_faltante"

    state["tree_protocol_id"] = None
    tree, _ = _verify(state)["contributions"]
    assert tree["status"] == "abstencion"

    state["tree_error"] = "Error tecnico: RuntimeError."
    tree, _ = _verify(state)["contributions"]
    assert tree["status"] == "error_tecnico"
