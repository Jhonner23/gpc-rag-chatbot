"""Pruebas del recorrido de los arboles clinicos aprobados (agents/flowchart.py)
y de la interpretacion de respuestas del wizard (agents/tree_agent.py).

Usan los YAML reales de agents/protocols/, sin modelos de lenguaje."""

from __future__ import annotations

from gpc_rag.agents.flowchart import UNKNOWN, Flowchart, walk
from gpc_rag.agents.protocol_registry import (
    available_protocol_ids,
    describe_available_protocols,
    get_flowchart,
)
from gpc_rag.agents.tree_agent import _phrase_question, _render_consult, parse_choice

_ITU = "CT-PL-197"
_NAC = "CT-PL-193"


def _all_routes(chart: Flowchart) -> list:
    """Recorre todas las combinaciones de respuestas posibles del arbol."""
    finished = []
    pending = [{}]
    while pending:
        answers = pending.pop()
        result = walk(chart, answers)
        if result.status == "pendiente":
            pending += [
                {**answers, result.stopped_at.id: option} for option in result.options
            ]
        else:
            finished.append(result)
    return finished


def test_registry_loads_both_approved_trees() -> None:
    assert available_protocol_ids() == [_NAC, _ITU]
    infos = {p.id: p for p in describe_available_protocols()}
    assert get_flowchart(_NAC).version == "v09"
    assert get_flowchart(_ITU).version == "v06"
    assert "neumonía" in infos[_NAC].name.lower()
    assert "ITU" in infos[_ITU].name


def test_every_itu_route_ends_in_a_final_node_or_a_known_abstention() -> None:
    routes = _all_routes(get_flowchart(_ITU))
    assert len(routes) > 20
    complete = [r for r in routes if r.status == "completo"]
    stuck = {r.stopped_at.id for r in routes if r.status == "sin_ruta"}
    # Unica rama sin salida del diagrama aprobado: sintomas que no cumplen
    # criterios de ITU alta ni baja llegan a "¿Paciente gestante?" sin
    # clasificacion; el agente se abstiene en vez de inferir una ruta.
    assert stuck == {"D_GEST"}
    assert all(
        r.steps[-1].node.kind in {"endpoint", "treatment", "test", "info"}
        for r in complete
    )


def _complete_with_first_options(chart: Flowchart, answers: dict):
    """Completa las decisiones pendientes con su primera opcion."""
    answers = dict(answers)
    result = walk(chart, answers)
    while result.status == "pendiente":
        answers[result.stopped_at.id] = result.options[0]
        result = walk(chart, answers)
    return result


def test_every_nac_branch_is_walkable_to_a_final_node() -> None:
    """NAC tiene demasiadas combinaciones para enumerarlas todas: se cubre
    cada opcion de cada decision al menos una vez."""
    chart = get_flowchart(_NAC)
    finals = set()
    covered = set()
    for node in chart.nodes.values():
        if not node.is_decision:
            continue
        route = {d.id: label for d, label in chart.route_to(node.id)}
        for edge in chart.out_edges(node.id):
            result = _complete_with_first_options(chart, {**route, node.id: edge.label})
            assert result.status == "completo"
            if any(
                s.node.id == node.id and s.choice == edge.label for s in result.steps
            ):
                covered.add((node.id, edge.label))
            finals.add(result.steps[-1].node.id)
    all_branches = {
        (n.id, e.label)
        for n in chart.nodes.values()
        if n.is_decision
        for e in chart.out_edges(n.id)
    }
    assert covered == all_branches
    assert finals <= {"no_confirmada", "ambulatorio", "alta", "continuar"}


def test_pregnancy_options_follow_the_previous_classification() -> None:
    chart = get_flowchart(_ITU)
    answers = {"D1": "Sí", "D_HIGH": "No", "D_LOW": "Sí", "D_CITO": "Sí", "D_UCX": "Sí"}
    result = walk(chart, answers)
    assert result.status == "pendiente"
    assert result.stopped_at.id == "D_GEST"
    assert result.options == ["Sí + ITU baja", "No + ITU baja"]


def test_conditional_continuations_are_notes_not_assumptions() -> None:
    chart = get_flowchart(_ITU)
    answers = {
        "D1": "Sí",
        "D_HIGH": "No",
        "D_LOW": "Sí",
        "D_CITO": "Sí",
        "D_UCX": "Sí",
        "D_GEST": "Sí + ITU baja",
    }
    result = walk(chart, answers)
    assert result.status == "completo"
    assert result.steps[-1].node.id == "LOW_G_TX"
    assert [(label, node.id) for label, node in result.notes] == [
        ("si recurrencia", "G_PREV"),
        ("si criterio de imagen", "G_IMG"),
    ]


def test_independent_evaluations_are_all_walked_before_discharge() -> None:
    chart = get_flowchart(_NAC)
    answers = {
        "neutropenia": "No",
        "infiltrado": "Sí",
        "hosp_dec": "Sí",
        "choque": "No",
        "desatura": "No",
        "sospecha_covid": "No",
        "muestra": "Sí",
        "hemo_dec": "No",
        "derrame": "No",
        "uci_dec": "No",
        "postviral": "No",
        "mrsa": "No",
        "psa_no_mrsa": "No",
        "virus_solo": "No",
        "influenza": "Sí",
    }
    result = walk(chart, answers)
    assert result.stopped_at.id == "covid"  # aun falta una de las tres evaluaciones
    result = walk(chart, {**answers, "covid": "No", "alta_dec": "Sí"})
    path = [s.node.id for s in result.steps]
    assert path.index("oseltamivir") < path.index("seguimiento") < path.index("alta")


def test_unknown_answer_stops_the_walk() -> None:
    result = walk(get_flowchart(_NAC), {"neutropenia": UNKNOWN})
    assert result.status == "dato_desconocido"
    assert result.stopped_at.id == "neutropenia"


def test_parse_choice() -> None:
    yes_no = ["Sí", "No"]
    assert parse_choice("si", yes_no) == "Sí"
    assert parse_choice("Sí.", yes_no) == "Sí"
    assert parse_choice("no", yes_no) == "No"
    assert parse_choice("2", yes_no) == "No"
    assert parse_choice("no sé", yes_no) == UNKNOWN
    assert parse_choice("quizas", yes_no) is None
    three = ["No", "Sí — gestante", "Sí — procedimiento"]
    assert parse_choice("si", three) is None  # ambiguo: se vuelve a preguntar
    assert parse_choice("si gestante", three) == "Sí — gestante"
    assert parse_choice("3", three) == "Sí — procedimiento"
    assert parse_choice("no", ["Cumple", "No cumple"]) == "No cumple"
    assert parse_choice("si", ["Cumple", "No cumple"]) == "Cumple"


def test_question_shows_supporting_criteria_and_options() -> None:
    chart = get_flowchart(_NAC)
    node = chart.nodes["hosp_dec"]
    text = _phrase_question(chart, node, ["No", "Sí"])
    assert "CURB-65" in text  # recuadro de criterios de hospitalizacion
    assert "No / Sí" in text
    assert "no sé" in text


def test_consult_mode_shows_route_and_other_branches() -> None:
    chart = get_flowchart(_ITU)
    result = _render_consult(chart, ["BA_NO_TX"])
    assert "NO TRATAR bacteriuria asintomática con antibiótico" in result["answer_text"]
    assert "Aplica si:" in result["answer_text"]
    assert (
        "GESTANTE + bacteriuria asintomática" in result["answer_text"]
    )  # otra rama de la misma decision
    assert all(c["page"] == 1 and c["verbatim_text"] for c in result["citations"])
    assert result["citations"][0]["source_file"].startswith(
        "Árbol clínico CT-PL-197 v06"
    )
