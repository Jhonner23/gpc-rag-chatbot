"""Pruebas del grafo de agentes (coordinador -> arbol || RAG -> evaluador ->
verificador -> integrador -> coordinador).

No llaman a Ollama real: se mockea `ollama.Client` en coordinator.py,
tree_agent.py, evaluator.py y verifier.py, y se usa un pipeline falso en vez de
QueryPipeline real (que requiere Qdrant/Ollama corriendo). El objetivo es
verificar el ENRUTAMIENTO del grafo (cuando se rechaza, cuando se reintenta,
cuando se agotan los reintentos, como se presentan las aportaciones del arbol
y del RAG, la falla cerrada del evaluador, y el wizard de
preguntas via interrupt()), no la calidad de los
prompts en si -- eso se valida a mano contra el chatbot real, como el resto
de la sesion.
"""

from __future__ import annotations

import json
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
_MAX_WIZARD_TURNS_IN_TEST = 25


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
    assert result.panels == []
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
    tree_panel, rag_panel = result.panels
    assert tree_panel.status == "abstención"
    assert tree_panel.content.startswith("El árbol de decisión no llegó a una respuesta")
    assert rag_panel.status == "respuesta fundamentada (verificada)"
    assert rag_panel.content.startswith("Respuesta bien fundamentada.")
    assert len(rag_panel.sources) == 1
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


# Respuestas simuladas del agente arbol: el clasificador elige NAC en modo
# caso y la extraccion inicial responde todas las decisiones de una ruta
# completa del arbol aprobado (NAC confirmada, hospitalizada en sala, sin
# factores de riesgo, influenza positiva, cumple criterios de alta), de modo
# que el recorrido llega al final sin preguntar nada.
_NAC_FULL_ROUTE = {
    "neutropenia": "No", "infiltrado": "Sí", "hosp_dec": "Sí", "choque": "No",
    "desatura": "No", "sospecha_covid": "No", "muestra": "Sí", "hemo_dec": "No",
    "derrame": "No", "uci_dec": "No", "postviral": "No", "mrsa": "No",
    "psa_no_mrsa": "No", "virus_solo": "No", "influenza": "Sí", "covid": "No",
    "alta_dec": "Sí",
}
_TREE_MATCH_RESPONSES = [
    _chat_response('{"protocol": "CT-PL-193", "modo": "caso"}'),
    _chat_response(json.dumps(_NAC_FULL_ROUTE)),
]
# El verificador usa el mismo cliente para la relacion del arbol y para las
# contradicciones: una sola respuesta simulada con ambas claves sirve para los dos.
_NO_CONTRADICTION = _chat_response('{"contradiccion": false, "relacionada": true}')
_TREE_QUESTION = "Paciente con sospecha de neumonia y cuadro clinico compatible, ¿que dice el protocolo?"


def test_tree_and_rag_are_shown_as_separate_verified_contributions() -> None:
    """Si el arbol recorre el diagrama y el RAG es aprobado, cada uno queda en
    su panel con sus fuentes. No hay sintesis redactada por un modelo: la
    respuesta de la API es la composicion literal en bloques."""
    pipeline = _FakePipeline(answers=["Respuesta del RAG con su cita."])

    with (
        patch("gpc_rag.agents.coordinator.Client") as mock_coord_client,
        patch("gpc_rag.agents.tree_agent.Client") as mock_tree_client,
        patch("gpc_rag.agents.evaluator.Client") as mock_eval_client,
        patch("gpc_rag.agents.verifier.Client") as mock_verif_client,
    ):
        mock_verif_client.return_value.chat.return_value = _NO_CONTRADICTION
        mock_coord_client.return_value.chat.return_value = _chat_response('{"verdict": "ADECUADA"}')
        mock_tree_client.return_value.chat.side_effect = list(_TREE_MATCH_RESPONSES)
        mock_eval_client.return_value.chat.return_value = _chat_response('{"verdict": "OK"}')

        result = run_agentic(_TREE_QUESTION, _CFG, pipeline)

    assert isinstance(result, RagAnswer)
    assert [p.status for p in result.panels] == [
        "respuesta fundamentada (verificada)",
        "respuesta fundamentada (verificada)",
    ]
    assert "árbol clínico CT-PL-193 v09" in result.panels[0].content
    assert result.panels[1].content == "Respuesta del RAG con su cita."
    assert result.panels[0].sources and result.panels[1].sources
    assert result.notes == []
    assert "Según el árbol clínico del protocolo institucional:" in result.answer
    assert "Según los documentos recuperados (RAG):" in result.answer
    assert {s.retrieval_method for s in result.sources} == {"arbol", "rerank"}


def test_tree_answer_is_published_alone_when_rag_is_rejected() -> None:
    """Si el evaluador rechaza al RAG (tras el reintento) pero el arbol llego
    a un resultado, se publica solo la aportacion del arbol."""
    pipeline = _FakePipeline(answers=["RAG mal fundamentado.", "RAG sigue mal."])

    with (
        patch("gpc_rag.agents.coordinator.Client") as mock_coord_client,
        patch("gpc_rag.agents.tree_agent.Client") as mock_tree_client,
        patch("gpc_rag.agents.evaluator.Client") as mock_eval_client,
        patch("gpc_rag.agents.verifier.Client") as mock_verif_client,
    ):
        mock_verif_client.return_value.chat.return_value = _NO_CONTRADICTION
        mock_coord_client.return_value.chat.return_value = _chat_response('{"verdict": "ADECUADA"}')
        mock_tree_client.return_value.chat.side_effect = list(_TREE_MATCH_RESPONSES)
        mock_eval_client.return_value.chat.return_value = _chat_response(
            '{"verdict": "RECHAZADA", "reason": "sin fundamento."}'
        )

        result = run_agentic(_TREE_QUESTION, _CFG, pipeline)

    assert "árbol clínico CT-PL-193 v09" in result.answer
    assert "RAG mal fundamentado" not in result.answer
    rag_panel = result.panels[1]
    assert rag_panel.status == "respuesta fundamentada (no publicada: rechazada por el evaluador)"
    assert "RAG mal fundamentado" not in rag_panel.content
    assert rag_panel.sources == []
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
    assert result.panels[1].status == "respuesta fundamentada (no publicada: no fue posible verificarla)"
    assert "Respuesta del RAG." not in result.panels[1].content
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
        patch("gpc_rag.agents.verifier.Client") as mock_verif_client,
    ):
        mock_verif_client.return_value.chat.return_value = _NO_CONTRADICTION
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
    # Respondiendo "si" a todo, el recorrido de NAC llega al final del arbol.
    assert result.panels[0].status == "respuesta fundamentada (verificada)"
    assert "árbol clínico CT-PL-193 v09" in result.panels[0].content


def test_contradiction_is_reported_as_a_verifier_note() -> None:
    """El verificador no resuelve una contradiccion entre arbol y RAG: la
    conserva y se informa como nota, junto a ambas aportaciones."""
    pipeline = _FakePipeline(answers=["Respuesta del RAG con su cita."])

    with (
        patch("gpc_rag.agents.coordinator.Client") as mock_coord_client,
        patch("gpc_rag.agents.tree_agent.Client") as mock_tree_client,
        patch("gpc_rag.agents.evaluator.Client") as mock_eval_client,
        patch("gpc_rag.agents.verifier.Client") as mock_verif_client,
    ):
        mock_coord_client.return_value.chat.return_value = _chat_response('{"verdict": "ADECUADA"}')
        mock_tree_client.return_value.chat.side_effect = list(_TREE_MATCH_RESPONSES)
        mock_eval_client.return_value.chat.return_value = _chat_response('{"verdict": "OK"}')
        mock_verif_client.return_value.chat.return_value = _chat_response(
            '{"contradiccion": true, "detalle": "difieren en el sitio de manejo.", "relacionada": true}'
        )

        result = run_agentic(_TREE_QUESTION, _CFG, pipeline)

    assert len(result.notes) == 1
    assert "difieren en el sitio de manejo" in result.notes[0]
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
        patch("gpc_rag.agents.verifier.Client") as mock_verif_client,
    ):
        mock_coord_client.return_value.chat.return_value = _chat_response('{"verdict": "ADECUADA"}')
        mock_tree_client.return_value.chat.side_effect = list(_TREE_MATCH_RESPONSES)
        mock_verif_client.return_value.chat.return_value = _NO_CONTRADICTION

        result = run_agentic(_TREE_QUESTION, _CFG, _BrokenPipeline())

    mock_eval_client.return_value.chat.assert_not_called()
    assert "árbol clínico CT-PL-193 v09" in result.answer
    assert result.panels[1].status == "error técnico"
    assert "error técnico" in result.panels[1].content


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
    assert result.panels[1].status == "error técnico"
    assert "error técnico" in result.panels[1].content
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
    assert result.panels[0].status == "error técnico"
    assert result.panels[0].content.startswith("El árbol de decisión no llegó a una respuesta")
    assert len(result.sources) == 1


def test_informative_question_uses_tree_consult_mode_without_wizard() -> None:
    """En modo consulta el arbol no pregunta nada: aporta las reglas del
    protocolo relacionadas con la pregunta (busqueda lexica sobre el paquete
    real), junto a la respuesta del RAG."""
    pipeline = _FakePipeline(answers=["Respuesta del RAG sobre bacteriuria."])

    with (
        patch("gpc_rag.agents.coordinator.Client") as mock_coord_client,
        patch("gpc_rag.agents.tree_agent.Client") as mock_tree_client,
        patch("gpc_rag.agents.evaluator.Client") as mock_eval_client,
        patch("gpc_rag.agents.verifier.Client") as mock_verif_client,
    ):
        mock_coord_client.return_value.chat.return_value = _chat_response('{"verdict": "ADECUADA"}')
        mock_tree_client.return_value.chat.side_effect = [
            _chat_response('{"protocol": "CT-PL-197", "modo": "consulta"}'),
            _chat_response('{"nodos": ["BA_NO_TX"]}'),  # seleccion de nodos
        ]
        mock_eval_client.return_value.chat.return_value = _chat_response('{"verdict": "OK"}')
        mock_verif_client.return_value.chat.return_value = _NO_CONTRADICTION

        result = run_agentic(
            "¿Qué recomiendan las guías sobre el manejo de la bacteriuria asintomática?",
            _CFG,
            pipeline,
        )

    assert isinstance(result, RagAnswer)  # nunca hubo pregunta de seguimiento
    assert mock_tree_client.return_value.chat.call_count == 2  # clasificacion + seleccion, sin wizard
    assert "NO TRATAR bacteriuria asintomática con antibiótico" in result.panels[0].content
    assert result.panels[1].content == "Respuesta del RAG sobre bacteriuria."


def test_unrelated_tree_contribution_is_not_published() -> None:
    """Si el verificador juzga que la aportacion del arbol no responde a la
    pregunta, no se publica y la respuesta sale del RAG."""
    pipeline = _FakePipeline(answers=["Respuesta del RAG con su cita."])

    with (
        patch("gpc_rag.agents.coordinator.Client") as mock_coord_client,
        patch("gpc_rag.agents.tree_agent.Client") as mock_tree_client,
        patch("gpc_rag.agents.evaluator.Client") as mock_eval_client,
        patch("gpc_rag.agents.verifier.Client") as mock_verif_client,
    ):
        mock_coord_client.return_value.chat.return_value = _chat_response('{"verdict": "ADECUADA"}')
        mock_tree_client.return_value.chat.side_effect = list(_TREE_MATCH_RESPONSES)
        mock_eval_client.return_value.chat.return_value = _chat_response('{"verdict": "OK"}')
        mock_verif_client.return_value.chat.return_value = _chat_response('{"relacionada": false}')

        result = run_agentic(_TREE_QUESTION, _CFG, pipeline)

    assert result.answer.startswith("Respuesta del RAG con su cita.")
    assert result.panels[0].status == "respuesta fundamentada (no publicada: no relacionada con la pregunta)"
    assert "no está relacionada con la pregunta" in result.panels[0].content
    assert all(s.retrieval_method == "rerank" for s in result.sources)
