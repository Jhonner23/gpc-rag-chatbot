"""Pruebas del grafo de agentes (coordinador -> RAG -> evaluador -> coordinador).

No llaman a Ollama real: se mockea `ollama.Client` en coordinator.py y
evaluator.py, y se usa un pipeline falso en vez de QueryPipeline real (que
requiere Qdrant/Ollama corriendo). El objetivo es verificar el ENRUTAMIENTO
del grafo (cuando se rechaza, cuando se reintenta, cuando se agotan los
reintentos), no la calidad de los prompts en si -- eso se valida a mano
contra el chatbot real, como el resto de la sesion.
"""

from __future__ import annotations

from unittest.mock import patch

from omegaconf import OmegaConf

from gpc_rag.agents.graph import run_agentic
from gpc_rag.common.models import Chunk, RagAnswer, RetrievedChunk

_CFG = OmegaConf.create(
    {
        "env": {"ollama_url": "http://localhost:11434"},
        "agents": {
            "coordinator_model": "fake-coordinator-model",
            "evaluator_model": "fake-evaluator-model",
        },
    }
)


def _chat_response(text: str) -> dict:
    return {"message": {"content": text}}


class _FakePipeline:
    """Sustituye a QueryPipeline: registra cuantas veces se le pregunto."""

    def __init__(self, answers: list[str]):
        self._answers = list(answers)
        self.calls: list[str | None] = []

    def answer(self, question: str, extra_instruction: str | None = None) -> RagAnswer:
        self.calls.append(extra_instruction)
        text = self._answers[len(self.calls) - 1]
        chunk = Chunk(text="contenido de la guia", source_file="guia.pdf", section="1.", page=1)
        sources = [RetrievedChunk(chunk=chunk, score=0.9, retrieval_method="rerank")]
        return RagAnswer(question=question, answer=text, sources=sources)


def test_coordinator_rejects_inappropriate_question_without_calling_rag():
    pipeline = _FakePipeline(answers=["no deberia llegar aqui"])

    with patch("gpc_rag.agents.coordinator.Client") as mock_client_cls:
        mock_client_cls.return_value.chat.return_value = _chat_response(
            '{"verdict": "RECHAZADA", "reason": "esa pregunta no es clinica."}'
        )
        result = run_agentic("Escribeme un poema", _CFG, pipeline)

    assert pipeline.calls == []  # el agente RAG nunca se invoco
    assert "no es clinica" in result.answer
    assert result.sources == []


def test_approves_and_returns_answer_with_sources_when_evaluator_ok():
    pipeline = _FakePipeline(answers=["Respuesta bien fundamentada."])

    with (
        patch("gpc_rag.agents.coordinator.Client") as mock_coord_client,
        patch("gpc_rag.agents.evaluator.Client") as mock_eval_client,
    ):
        mock_coord_client.return_value.chat.return_value = _chat_response('{"verdict": "ADECUADA"}')
        mock_eval_client.return_value.chat.return_value = _chat_response('{"verdict": "OK"}')

        result = run_agentic("¿Cual es el tratamiento de la NAC?", _CFG, pipeline)

    assert len(pipeline.calls) == 1
    assert pipeline.calls[0] is None  # sin instruccion de reintento
    assert result.answer == "Respuesta bien fundamentada."
    assert len(result.sources) == 1


def test_retries_once_with_stricter_instruction_then_succeeds():
    pipeline = _FakePipeline(answers=["Primera respuesta (mal fundamentada).", "Segunda respuesta (corregida)."])

    with (
        patch("gpc_rag.agents.coordinator.Client") as mock_coord_client,
        patch("gpc_rag.agents.evaluator.Client") as mock_eval_client,
    ):
        mock_coord_client.return_value.chat.return_value = _chat_response('{"verdict": "ADECUADA"}')
        mock_eval_client.return_value.chat.side_effect = [
            _chat_response(
                '{"verdict": "RECHAZADA", "reason": "menciona un farmaco que no esta en el contexto."}'
            ),
            _chat_response('{"verdict": "OK"}'),
        ]

        result = run_agentic("¿Cual es el tratamiento de la NAC?", _CFG, pipeline)

    assert len(pipeline.calls) == 2
    assert pipeline.calls[0] is None
    assert pipeline.calls[1] is not None
    assert "farmaco que no esta en el contexto" in pipeline.calls[1]
    assert result.answer == "Segunda respuesta (corregida)."
    assert len(result.sources) == 1


def test_falls_back_without_sources_when_evaluator_rejects_after_retry():
    pipeline = _FakePipeline(answers=["Primera respuesta.", "Segunda respuesta (sigue mal)."])

    with (
        patch("gpc_rag.agents.coordinator.Client") as mock_coord_client,
        patch("gpc_rag.agents.evaluator.Client") as mock_eval_client,
    ):
        mock_coord_client.return_value.chat.return_value = _chat_response('{"verdict": "ADECUADA"}')
        mock_eval_client.return_value.chat.return_value = _chat_response(
            '{"verdict": "RECHAZADA", "reason": "sigue sin fundamento."}'
        )

        result = run_agentic("¿Cual es el tratamiento de la NAC?", _CFG, pipeline)

    assert len(pipeline.calls) == 2  # 1 original + 1 reintento, no mas (MAX_RETRIES=1)
    assert "No puedo confirmar" in result.answer
    assert result.sources == []


def test_evaluator_short_circuits_on_well_formed_refusal_without_calling_llm():
    """Si el agente RAG ya dice explicitamente que no hay informacion
    suficiente, el evaluador debe aprobarla sin siquiera llamar al LLM
    juez (ver common/refusal.py) -- ese es justo el caso que el juez local
    fallaba en reconocer de forma confiable (ver diagnostico de la sesion,
    caso tuberculosis)."""
    pipeline = _FakePipeline(
        answers=["No encontré información suficiente en las guías para responder esta pregunta."]
    )

    with (
        patch("gpc_rag.agents.coordinator.Client") as mock_coord_client,
        patch("gpc_rag.agents.evaluator.Client") as mock_eval_client,
    ):
        mock_coord_client.return_value.chat.return_value = _chat_response('{"verdict": "ADECUADA"}')

        result = run_agentic("¿Cual es el tratamiento de la tuberculosis?", _CFG, pipeline)

    mock_eval_client.return_value.chat.assert_not_called()
    assert len(pipeline.calls) == 1
    assert "No encontré información suficiente" in result.answer
