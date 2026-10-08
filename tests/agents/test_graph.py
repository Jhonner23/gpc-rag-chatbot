"""Pruebas del grafo de agentes (coordinador -> arbol -> RAG -> evaluador ->
comparador -> coordinador).

No llaman a Ollama real: se mockea `ollama.Client` en coordinator.py,
tree_agent.py y evaluator.py, y se usa un pipeline falso en vez de
QueryPipeline real (que requiere Qdrant/Ollama corriendo). El objetivo es
verificar el ENRUTAMIENTO del grafo (cuando se rechaza, cuando se reintenta,
cuando se agotan los reintentos, cuando el arbol gana vs. cuando gana el
RAG, y el wizard de preguntas via interrupt()), no la calidad de los
prompts en si -- eso se valida a mano contra el chatbot real, como el resto
de la sesion.
"""

from __future__ import annotations

from unittest.mock import patch

from langgraph.checkpoint.memory import MemorySaver
from omegaconf import OmegaConf

from gpc_rag.agents.graph import run_agentic
from gpc_rag.common.models import Chunk, PendingTreeQuestion, RagAnswer, RetrievedChunk

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


# Respuesta por defecto del "clasificador de protocolo" del agente arbol:
# ningun protocolo aplica, para que estas pruebas de enrutamiento general
# se comporten exactamente igual que antes de agregar el agente arbol (el
# comparador cae directo al RAG).
_NO_PROTOCOL = _chat_response('{"protocol": "NINGUNO"}')
_MAX_WIZARD_TURNS_IN_TEST = 10


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


def test_coordinator_rejects_inappropriate_question_without_calling_rag() -> None:
    pipeline = _FakePipeline(answers=["no deberia llegar aqui"])

    with patch("gpc_rag.agents.coordinator.Client") as mock_client_cls:
        mock_client_cls.return_value.chat.return_value = _chat_response(
            '{"verdict": "RECHAZADA", "reason": "esa pregunta no es clinica."}'
        )
        result = run_agentic("Escribeme un poema", _CFG, pipeline)

    assert pipeline.calls == []  # el agente RAG nunca se invoco
    assert "no es clinica" in result.answer
    assert result.sources == []


def test_approves_and_returns_answer_with_sources_when_evaluator_ok() -> None:
    pipeline = _FakePipeline(answers=["Respuesta bien fundamentada."])

    with (
        patch("gpc_rag.agents.coordinator.Client") as mock_coord_client,
        patch("gpc_rag.agents.tree_agent.Client") as mock_tree_client,
        patch("gpc_rag.agents.evaluator.Client") as mock_eval_client,
    ):
        mock_coord_client.return_value.chat.return_value = _chat_response('{"verdict": "ADECUADA"}')
        mock_tree_client.return_value.chat.return_value = _NO_PROTOCOL
        mock_eval_client.return_value.chat.return_value = _chat_response('{"verdict": "OK"}')

        result = run_agentic("¿Cual es el tratamiento de la NAC?", _CFG, pipeline)

    assert len(pipeline.calls) == 1
    assert pipeline.calls[0] is None  # sin instruccion de reintento
    assert result.answer == "Respuesta bien fundamentada."
    assert len(result.sources) == 1


def test_retries_once_with_stricter_instruction_then_succeeds() -> None:
    pipeline = _FakePipeline(
        answers=["Primera respuesta (mal fundamentada).", "Segunda respuesta (corregida)."]
    )

    with (
        patch("gpc_rag.agents.coordinator.Client") as mock_coord_client,
        patch("gpc_rag.agents.tree_agent.Client") as mock_tree_client,
        patch("gpc_rag.agents.evaluator.Client") as mock_eval_client,
    ):
        mock_coord_client.return_value.chat.return_value = _chat_response('{"verdict": "ADECUADA"}')
        mock_tree_client.return_value.chat.return_value = _NO_PROTOCOL
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


def test_falls_back_without_sources_when_evaluator_rejects_after_retry() -> None:
    pipeline = _FakePipeline(answers=["Primera respuesta.", "Segunda respuesta (sigue mal)."])

    with (
        patch("gpc_rag.agents.coordinator.Client") as mock_coord_client,
        patch("gpc_rag.agents.tree_agent.Client") as mock_tree_client,
        patch("gpc_rag.agents.evaluator.Client") as mock_eval_client,
    ):
        mock_coord_client.return_value.chat.return_value = _chat_response('{"verdict": "ADECUADA"}')
        mock_tree_client.return_value.chat.return_value = _NO_PROTOCOL
        mock_eval_client.return_value.chat.return_value = _chat_response(
            '{"verdict": "RECHAZADA", "reason": "sigue sin fundamento."}'
        )

        result = run_agentic("¿Cual es el tratamiento de la NAC?", _CFG, pipeline)

    assert len(pipeline.calls) == 2  # 1 original + 1 reintento, no mas (MAX_RETRIES=1)
    assert "No puedo confirmar" in result.answer
    assert result.sources == []


def test_evaluator_short_circuits_on_well_formed_refusal_without_calling_llm() -> None:
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
        patch("gpc_rag.agents.tree_agent.Client") as mock_tree_client,
        patch("gpc_rag.agents.evaluator.Client") as mock_eval_client,
    ):
        mock_coord_client.return_value.chat.return_value = _chat_response('{"verdict": "ADECUADA"}')
        mock_tree_client.return_value.chat.return_value = _NO_PROTOCOL

        result = run_agentic("¿Cual es el tratamiento de la tuberculosis?", _CFG, pipeline)

    mock_eval_client.return_value.chat.assert_not_called()
    assert len(pipeline.calls) == 1
    assert "No encontré información suficiente" in result.answer


def test_tree_match_is_preferred_over_rag_answer() -> None:
    """Si el agente arbol llega a una regla MATCHED auditada (EXTRACTED),
    el comparador debe preferirla sobre la respuesta del RAG, aunque el RAG
    tambien haya sido aprobada por el evaluador."""
    pipeline = _FakePipeline(answers=["Respuesta generica del RAG (no deberia usarse)."])

    # El extractor del agente arbol devuelve, de una vez, datos suficientes
    # para que una regla real del protocolo NAC quede MATCHED sin tener que
    # preguntar nada (misma combinacion que ya se probo manualmente contra
    # el motor real: sospecha_neumonia + cuadro_clinico_compatible).
    tree_extraction = _chat_response(
        '{"sospecha_neumonia": true, "cuadro_clinico_compatible": true}'
    )

    with (
        patch("gpc_rag.agents.coordinator.Client") as mock_coord_client,
        patch("gpc_rag.agents.tree_agent.Client") as mock_tree_client,
        patch("gpc_rag.agents.evaluator.Client") as mock_eval_client,
    ):
        mock_coord_client.return_value.chat.return_value = _chat_response('{"verdict": "ADECUADA"}')
        mock_tree_client.return_value.chat.side_effect = [
            _chat_response('{"protocol": "CT-PL-193"}'),
            tree_extraction,
        ]
        mock_eval_client.return_value.chat.return_value = _chat_response('{"verdict": "OK"}')

        result = run_agentic(
            "Paciente con sospecha de neumonia y cuadro clinico compatible, ¿que dice el protocolo?",
            _CFG,
            pipeline,
        )

    assert isinstance(result, RagAnswer)
    assert "arbol de decision del protocolo CT-PL-193" in result.answer
    assert result.answer != "Respuesta generica del RAG (no deberia usarse)."
    assert all(s.retrieval_method == "arbol" for s in result.sources)


def test_tree_wizard_pauses_and_resumes_with_interrupt() -> None:
    """Si el agente arbol no tiene suficientes datos, el grafo debe pausarse
    (PendingTreeQuestion) y, al reanudar con la respuesta del usuario para
    el mismo thread_id, continuar hasta llegar a una conclusion."""
    pipeline = _FakePipeline(answers=["Respuesta generica del RAG (no deberia usarse)."])
    checkpointer = MemorySaver()

    with (
        patch("gpc_rag.agents.coordinator.Client") as mock_coord_client,
        patch("gpc_rag.agents.tree_agent.Client") as mock_tree_client,
        patch("gpc_rag.agents.evaluator.Client") as mock_eval_client,
    ):
        mock_coord_client.return_value.chat.return_value = _chat_response('{"verdict": "ADECUADA"}')
        mock_eval_client.return_value.chat.return_value = _chat_response('{"verdict": "OK"}')
        # Protocolo NAC, pero la extraccion inicial no saca ningun dato ->
        # el agente arbol tiene que empezar a preguntar.
        mock_tree_client.return_value.chat.side_effect = [
            _chat_response('{"protocol": "CT-PL-193"}'),
            _chat_response("{}"),
        ]

        first = run_agentic(
            "¿Que dice el protocolo sobre un paciente con neumonia?",
            _CFG,
            pipeline,
            checkpointer=checkpointer,
        )
        assert isinstance(first, PendingTreeQuestion)
        assert first.question

        # Se responde "si" a la primera pregunta del wizard, una y otra vez,
        # hasta que el grafo deje de pausarse (encuentra un MATCH o agota
        # el limite de preguntas). No importa cual variable especifica es
        # en cada paso -- lo que se prueba es el mecanismo de pausa/resume.
        result = first
        answers_sent = 0
        while isinstance(result, PendingTreeQuestion) and answers_sent < _MAX_WIZARD_TURNS_IN_TEST:
            result = run_agentic(
                "",
                _CFG,
                pipeline,
                checkpointer=checkpointer,
                thread_id=result.thread_id,
                resume_answer="si",
            )
            answers_sent += 1

    assert isinstance(result, RagAnswer)
    assert answers_sent > 0
