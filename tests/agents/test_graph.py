"""Pruebas del grafo de agentes (coordinador -> arbol || RAG -> evaluador ->
verificador -> integrador -> coordinador).

No llaman a Ollama real: se mockea `ollama.Client` en coordinator.py,
tree_agent.py, evaluator.py e integrator.py, y se usa un pipeline falso en vez de
QueryPipeline real (que requiere Qdrant/Ollama corriendo). El objetivo es
verificar el ENRUTAMIENTO del grafo (cuando se rechaza, cuando se reintenta,
cuando se agotan los reintentos, como se integran las aportaciones del arbol
y del RAG, la falla cerrada del evaluador y del integrador, y el wizard de
preguntas via interrupt()), no la calidad de los
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
# integrador publica solo la aportacion del RAG).
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
    assert result.answer.startswith("Respuesta bien fundamentada.")
    assert "árbol de decisión, abstención; RAG, respuesta fundamentada (verificada)" in result.answer
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
    assert result.answer.startswith("Segunda respuesta (corregida).")
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


# El extractor del agente arbol devuelve, de una vez, datos suficientes para
# que una regla real del protocolo NAC quede MATCHED sin tener que preguntar
# nada (misma combinacion que ya se probo manualmente contra el motor real:
# sospecha_neumonia + cuadro_clinico_compatible).
_TREE_MATCH_RESPONSES = [
    _chat_response('{"protocol": "CT-PL-193"}'),
    _chat_response('{"sospecha_neumonia": true, "cuadro_clinico_compatible": true}'),
]
_NO_CONTRADICTION = _chat_response('{"contradiccion": false}')
_TREE_QUESTION = "Paciente con sospecha de neumonia y cuadro clinico compatible, ¿que dice el protocolo?"


def test_tree_and_rag_are_integrated_into_single_answer() -> None:
    """Si el arbol llega a una regla auditada y el RAG es aprobado, el
    integrador redacta UNA respuesta consolidada y, si supera la verificacion
    de fidelidad, esa es la que se publica, con las fuentes de ambos agentes."""
    pipeline = _FakePipeline(answers=["Respuesta del RAG con su cita."])

    with (
        patch("gpc_rag.agents.coordinator.Client") as mock_coord_client,
        patch("gpc_rag.agents.tree_agent.Client") as mock_tree_client,
        patch("gpc_rag.agents.evaluator.Client") as mock_eval_client,
        patch("gpc_rag.agents.integrator.Client") as mock_integ_client,
        patch("gpc_rag.agents.verifier.Client") as mock_verif_client,
    ):
        mock_verif_client.return_value.chat.return_value = _NO_CONTRADICTION
        mock_coord_client.return_value.chat.return_value = _chat_response('{"verdict": "ADECUADA"}')
        mock_tree_client.return_value.chat.side_effect = list(_TREE_MATCH_RESPONSES)
        mock_eval_client.return_value.chat.return_value = _chat_response('{"verdict": "OK"}')
        mock_integ_client.return_value.chat.side_effect = [
            _chat_response("Respuesta consolidada del protocolo CT-PL-193 y del RAG."),
            _chat_response('{"verdict": "OK"}'),
        ]

        result = run_agentic(_TREE_QUESTION, _CFG, pipeline)

    assert isinstance(result, RagAnswer)
    assert result.answer.startswith("Respuesta consolidada del protocolo CT-PL-193 y del RAG.")
    assert "árbol de decisión, respuesta fundamentada (verificada); RAG, respuesta fundamentada (verificada)" in result.answer
    methods = {s.retrieval_method for s in result.sources}
    assert methods == {"arbol", "rerank"}
    assert mock_integ_client.return_value.chat.call_count == 2  # sintesis + fidelidad


def test_integrator_falls_back_to_blocks_when_fidelity_check_fails() -> None:
    """Falla cerrada del integrador: si la sintesis no supera la verificacion
    de fidelidad, se publica la composicion determinista (bloques)."""
    pipeline = _FakePipeline(answers=["Respuesta del RAG con su cita."])

    with (
        patch("gpc_rag.agents.coordinator.Client") as mock_coord_client,
        patch("gpc_rag.agents.tree_agent.Client") as mock_tree_client,
        patch("gpc_rag.agents.evaluator.Client") as mock_eval_client,
        patch("gpc_rag.agents.integrator.Client") as mock_integ_client,
        patch("gpc_rag.agents.verifier.Client") as mock_verif_client,
    ):
        mock_verif_client.return_value.chat.return_value = _NO_CONTRADICTION
        mock_coord_client.return_value.chat.return_value = _chat_response('{"verdict": "ADECUADA"}')
        mock_tree_client.return_value.chat.side_effect = list(_TREE_MATCH_RESPONSES)
        mock_eval_client.return_value.chat.return_value = _chat_response('{"verdict": "OK"}')
        mock_integ_client.return_value.chat.side_effect = [
            _chat_response("Sintesis que agrega una dosis inventada."),
            _chat_response('{"verdict": "RECHAZADA", "reason": "agrega una dosis."}'),
        ]

        result = run_agentic(_TREE_QUESTION, _CFG, pipeline)

    assert "dosis inventada" not in result.answer
    assert "Según el protocolo institucional (árbol de decisión)" in result.answer
    assert "arbol de decision del protocolo CT-PL-193" in result.answer
    assert "Según los documentos recuperados (RAG)" in result.answer
    assert "Respuesta del RAG con su cita." in result.answer


def test_integrator_falls_back_to_blocks_when_synthesis_model_fails() -> None:
    """Un fallo del modelo durante la sintesis no tumba la respuesta: se
    publica la composicion determinista."""
    pipeline = _FakePipeline(answers=["Respuesta del RAG con su cita."])

    with (
        patch("gpc_rag.agents.coordinator.Client") as mock_coord_client,
        patch("gpc_rag.agents.tree_agent.Client") as mock_tree_client,
        patch("gpc_rag.agents.evaluator.Client") as mock_eval_client,
        patch("gpc_rag.agents.integrator.Client") as mock_integ_client,
        patch("gpc_rag.agents.verifier.Client") as mock_verif_client,
    ):
        mock_verif_client.return_value.chat.return_value = _NO_CONTRADICTION
        mock_coord_client.return_value.chat.return_value = _chat_response('{"verdict": "ADECUADA"}')
        mock_tree_client.return_value.chat.side_effect = list(_TREE_MATCH_RESPONSES)
        mock_eval_client.return_value.chat.return_value = _chat_response('{"verdict": "OK"}')
        mock_integ_client.return_value.chat.side_effect = ConnectionError("Ollama caido")

        result = run_agentic(_TREE_QUESTION, _CFG, pipeline)

    assert "Según el protocolo institucional (árbol de decisión)" in result.answer
    assert "Según los documentos recuperados (RAG)" in result.answer


def test_tree_answer_is_published_alone_when_rag_is_rejected() -> None:
    """Si el evaluador rechaza al RAG (tras el reintento) pero el arbol llego
    a una regla auditada, se publica solo la aportacion del arbol, sin
    llamar al integrador LLM."""
    pipeline = _FakePipeline(answers=["RAG mal fundamentado.", "RAG sigue mal."])

    with (
        patch("gpc_rag.agents.coordinator.Client") as mock_coord_client,
        patch("gpc_rag.agents.tree_agent.Client") as mock_tree_client,
        patch("gpc_rag.agents.evaluator.Client") as mock_eval_client,
        patch("gpc_rag.agents.integrator.Client") as mock_integ_client,
        patch("gpc_rag.agents.verifier.Client") as mock_verif_client,
    ):
        mock_verif_client.return_value.chat.return_value = _NO_CONTRADICTION
        mock_coord_client.return_value.chat.return_value = _chat_response('{"verdict": "ADECUADA"}')
        mock_tree_client.return_value.chat.side_effect = list(_TREE_MATCH_RESPONSES)
        mock_eval_client.return_value.chat.return_value = _chat_response(
            '{"verdict": "RECHAZADA", "reason": "sin fundamento."}'
        )

        result = run_agentic(_TREE_QUESTION, _CFG, pipeline)

    mock_integ_client.return_value.chat.assert_not_called()
    assert "arbol de decision del protocolo CT-PL-193" in result.answer
    assert "RAG mal fundamentado" not in result.answer
    assert "RAG, respuesta fundamentada (no publicada: rechazada por el evaluador)" in result.answer
    assert all(s.retrieval_method == "arbol" for s in result.sources)


def test_unreadable_evaluator_verdict_fails_closed_without_regenerating_rag() -> None:
    """Falla cerrada del evaluador: si el veredicto no es interpretable ni
    tras reintentar el juicio, la respuesta no se publica y el RAG NO se
    regenera (el problema es del juez, no de la respuesta)."""
    pipeline = _FakePipeline(answers=["Respuesta del RAG."])

    with (
        patch("gpc_rag.agents.coordinator.Client") as mock_coord_client,
        patch("gpc_rag.agents.tree_agent.Client") as mock_tree_client,
        patch("gpc_rag.agents.evaluator.Client") as mock_eval_client,
    ):
        mock_coord_client.return_value.chat.return_value = _chat_response('{"verdict": "ADECUADA"}')
        mock_tree_client.return_value.chat.return_value = _NO_PROTOCOL
        mock_eval_client.return_value.chat.return_value = _chat_response("esto no es json")

        result = run_agentic("¿Cual es el tratamiento de la NAC?", _CFG, pipeline)

    assert len(pipeline.calls) == 1
    assert mock_eval_client.return_value.chat.call_count == 2  # juicio + 1 reintento
    assert "No puedo confirmar" in result.answer
    assert "no publicada: no fue posible verificarla" in result.answer
    assert result.sources == []


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
        patch("gpc_rag.agents.integrator.Client") as mock_integ_client,
        patch("gpc_rag.agents.verifier.Client") as mock_verif_client,
    ):
        mock_verif_client.return_value.chat.return_value = _NO_CONTRADICTION
        mock_coord_client.return_value.chat.return_value = _chat_response('{"verdict": "ADECUADA"}')
        mock_eval_client.return_value.chat.return_value = _chat_response('{"verdict": "OK"}')
        mock_integ_client.return_value.chat.return_value = _chat_response('{"verdict": "OK"}')
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


def test_contradiction_is_reported_to_synthesis_and_in_answer() -> None:
    """El verificador no resuelve una contradiccion entre arbol y RAG: la
    conserva, la pasa a la sintesis y la informa en la respuesta."""
    pipeline = _FakePipeline(answers=["Respuesta del RAG con su cita."])

    with (
        patch("gpc_rag.agents.coordinator.Client") as mock_coord_client,
        patch("gpc_rag.agents.tree_agent.Client") as mock_tree_client,
        patch("gpc_rag.agents.evaluator.Client") as mock_eval_client,
        patch("gpc_rag.agents.integrator.Client") as mock_integ_client,
        patch("gpc_rag.agents.verifier.Client") as mock_verif_client,
    ):
        mock_coord_client.return_value.chat.return_value = _chat_response('{"verdict": "ADECUADA"}')
        mock_tree_client.return_value.chat.side_effect = list(_TREE_MATCH_RESPONSES)
        mock_eval_client.return_value.chat.return_value = _chat_response('{"verdict": "OK"}')
        mock_verif_client.return_value.chat.return_value = _chat_response(
            '{"contradiccion": true, "detalle": "difieren en el sitio de manejo."}'
        )
        mock_integ_client.return_value.chat.side_effect = [
            _chat_response("Respuesta consolidada que menciona la diferencia."),
            _chat_response('{"verdict": "OK"}'),
        ]

        result = run_agentic(_TREE_QUESTION, _CFG, pipeline)

    synthesis_prompt = mock_integ_client.return_value.chat.call_args_list[0].kwargs["messages"][1]["content"]
    assert "NOTAS DEL VERIFICADOR" in synthesis_prompt
    assert "difieren en el sitio de manejo" in synthesis_prompt
    assert "Posible contradiccion" in result.answer


def test_rag_technical_error_does_not_break_the_answer() -> None:
    """Si la rama del RAG falla, queda en estado de error tecnico y la
    respuesta del arbol se publica igual (no hay error 503)."""

    class _BrokenPipeline:
        def answer(self, question: str, extra_instruction: str | None = None) -> RagAnswer:
            raise RuntimeError("Qdrant caido")

    with (
        patch("gpc_rag.agents.coordinator.Client") as mock_coord_client,
        patch("gpc_rag.agents.tree_agent.Client") as mock_tree_client,
        patch("gpc_rag.agents.evaluator.Client") as mock_eval_client,
    ):
        mock_coord_client.return_value.chat.return_value = _chat_response('{"verdict": "ADECUADA"}')
        mock_tree_client.return_value.chat.side_effect = list(_TREE_MATCH_RESPONSES)

        result = run_agentic(_TREE_QUESTION, _CFG, _BrokenPipeline())

    mock_eval_client.return_value.chat.assert_not_called()
    assert "arbol de decision del protocolo CT-PL-193" in result.answer
    assert "RAG, error técnico" in result.answer


def test_rag_branch_timeout_is_reported_as_technical_error() -> None:
    """Si la rama del RAG supera su tiempo maximo, su resultado tardio se
    descarta y la rama queda en error tecnico."""
    import time

    class _SlowPipeline(_FakePipeline):
        def answer(self, question: str, extra_instruction: str | None = None) -> RagAnswer:
            time.sleep(1.0)
            return super().answer(question, extra_instruction)

    cfg = OmegaConf.merge(_CFG, {"agents": {"rag_timeout_s": 0.2}})
    pipeline = _SlowPipeline(answers=["Respuesta tardia."])

    with (
        patch("gpc_rag.agents.coordinator.Client") as mock_coord_client,
        patch("gpc_rag.agents.tree_agent.Client") as mock_tree_client,
    ):
        mock_coord_client.return_value.chat.return_value = _chat_response('{"verdict": "ADECUADA"}')
        mock_tree_client.return_value.chat.return_value = _NO_PROTOCOL

        result = run_agentic("¿Cual es el tratamiento de la NAC?", cfg, pipeline)

    assert "Respuesta tardia" not in result.answer
    assert "No puedo confirmar" in result.answer
    assert "RAG, error técnico" in result.answer
    assert result.sources == []


def test_tree_technical_error_does_not_break_the_answer() -> None:
    """Si la rama del arbol falla, queda en error tecnico y la respuesta del
    RAG se publica igual."""
    pipeline = _FakePipeline(answers=["Respuesta bien fundamentada."])

    with (
        patch("gpc_rag.agents.coordinator.Client") as mock_coord_client,
        patch("gpc_rag.agents.tree_agent.Client") as mock_tree_client,
        patch("gpc_rag.agents.evaluator.Client") as mock_eval_client,
    ):
        mock_coord_client.return_value.chat.return_value = _chat_response('{"verdict": "ADECUADA"}')
        mock_tree_client.return_value.chat.side_effect = ConnectionError("Ollama caido")
        mock_eval_client.return_value.chat.return_value = _chat_response('{"verdict": "OK"}')

        result = run_agentic("¿Cual es el tratamiento de la NAC?", _CFG, pipeline)

    assert result.answer.startswith("Respuesta bien fundamentada.")
    assert "árbol de decisión, error técnico" in result.answer
    assert len(result.sources) == 1
