"""Agente arbol: recorre el arbol clinico aprobado por el comite de la clinica
(diagrama de flujo, ver agents/flowchart.py).

Flujo (dentro del grafo, ver agents/graph.py):

1. ``tree_prepare_node`` (nodo normal, corre una sola vez): decide que
   protocolo aplica y el modo de la pregunta. En modo caso, un modelo local
   responde las preguntas del arbol que el texto del usuario permita
   responder explicitamente; las demas quedan como desconocidas.
2. ``tree_agent_node`` (nodo con `interrupt()`):
   - Modo consulta: devuelve los nodos del arbol relacionados con la
     pregunta, con la ruta que lleva a cada uno y las otras ramas de su
     decision. No pregunta nada.
   - Modo caso: recorre el arbol desde el inicio. En cada decision sin
     respuesta pregunta al usuario (una a la vez, estilo wizard) mostrando
     las opciones del diagrama y los criterios de apoyo. El recorrido termina
     en un nodo final del diagrama, o se detiene si falta un dato
     (informacion_faltante) o si ninguna rama aplica al caso
     (ausencia_de_coincidencias): el agente no infiere rutas que el diagrama
     no describe.

La aportacion cita el texto literal de los nodos recorridos, del arbol
aprobado (archivo, seccion o pagina del protocolo indicada en el diagrama).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langgraph.types import interrupt
from ollama import Client
from omegaconf import DictConfig

from gpc_rag.agents.flowchart import (
    UNKNOWN,
    Flowchart,
    FlowNode,
    Step,
    WalkResult,
    decision_overview,
    initial_section,
    normalize,
    rank_nodes_for_query,
    walk,
)
from gpc_rag.agents.protocol_registry import (
    available_protocol_ids,
    describe_available_protocols,
    get_flowchart,
)
from gpc_rag.agents.schemas import (
    decisions_schema,
    node_selection_schema,
    protocol_schema,
)
from gpc_rag.agents.state import AgentState
from gpc_rag.agents.timeouts import llm_timeout
from gpc_rag.common.llm_options import ollama_options

logger = logging.getLogger(__name__)

# Tope de preguntas al usuario por conversacion. El recorrido completo de NAC
# tiene hasta ~17 decisiones; las que el texto ya responde no se preguntan.
MAX_QUESTIONS = 20

_LLM_UNKNOWN = "DESCONOCIDO"

_YES_WORDS = {"si", "s", "yes", "y", "afirmativo", "positivo", "verdadero", "cumple"}
_NO_WORDS = {"no", "n", "negativo", "falso"}
_UNKNOWN_RE = re.compile(r"^(no se|nose|no lo se|no sabe|ns|desconoc\w*|sin dato\w*)\b")

_EXTRACT_SYSTEM_PROMPT_TEMPLATE = """\
Eres un asistente que lee la descripcion de un caso clinico y responde \
preguntas de un arbol de decision institucional. Responde cada pregunta SOLO \
si el texto del usuario lo dice explicitamente; si no lo menciona o hay \
dudas, responde {unknown}. No uses conocimiento propio ni supongas datos.

Preguntas (id: pregunta [opciones]):
{catalog}

Responde UNICAMENTE con un objeto JSON {{"<id>": "<opcion exacta o {unknown}>", ...}} \
con todas las ids.
"""


# Clasificacion del protocolo y del modo ------------------------------------


def _build_protocol_select_prompt() -> str:
    """Prompt del clasificador a partir de los protocolos disponibles."""
    lines = [
        (
            "Eres un clasificador que decide a cual protocolo clinico "
            "institucional pertenece una pregunta, entre los siguientes "
            "(identificados por su id exacto):"
        ),
        "",
    ]
    for protocol in describe_available_protocols():
        lines.append(f"- {protocol.id}: {protocol.name} ({protocol.description})")
    lines += [
        "",
        "Si la pregunta no corresponde claramente a ninguno de estos protocolos, responde NINGUNO.",
        "",
        (
            "Ademas, clasifica el MODO de la pregunta:\n"
            "- caso: describe a un paciente concreto o pide decidir para un caso "
            "(ej. 'paciente de 70 años con neumonia y derrame, ¿se hospitaliza?').\n"
            "- consulta: pide informacion general del protocolo, sin datos de un "
            "paciente (ej. '¿que recomienda la guia sobre...?', '¿cuando se trata...?').\n"
            "Si tienes dudas, usa caso."
        ),
        "",
        (
            "Responde UNICAMENTE con un objeto JSON con las claves protocol y modo, "
            'por ejemplo {"protocol": "<id o NINGUNO>", "modo": "caso"}.'
        ),
    ]
    return "\n".join(lines)


def classify_question(question: str, cfg: DictConfig) -> tuple[str | None, str]:
    """Devuelve (id del protocolo o None, modo "caso" | "consulta")."""
    valid_ids = available_protocol_ids()
    if not valid_ids:
        logger.warning("Agente arbol: no hay arboles en agents/protocols/.")
        return None, "caso"

    client = Client(host=cfg.env.ollama_url, timeout=llm_timeout(cfg))
    response = client.chat(
        model=cfg.agents.coordinator_model,
        messages=[
            {"role": "system", "content": _build_protocol_select_prompt()},
            {"role": "user", "content": question},
        ],
        format=protocol_schema(valid_ids),
        options=ollama_options(cfg, 50),
        stream=False,
    )
    content = response["message"]["content"].strip()
    try:
        parsed = json.loads(content)
        protocol_id = str(parsed.get("protocol", "")).strip().upper()
        mode = str(parsed.get("modo", "caso")).strip().lower()
    except (json.JSONDecodeError, AttributeError):
        logger.warning(
            "Agente arbol: no se pudo parsear la seleccion de protocolo (%r)", content
        )
        return None, "caso"
    mode = mode if mode in {"caso", "consulta"} else "caso"
    return (protocol_id if protocol_id in valid_ids else None), mode


def select_protocol(question: str, cfg: DictConfig) -> str | None:
    """Compatibilidad: solo el id del protocolo detectado (o None)."""
    return classify_question(question, cfg)[0]


# Respuestas a las decisiones -----------------------------------------------


def _headline(text: str) -> str:
    """Primeras lineas de un nodo, antes de su lista de viñetas."""
    head = []
    for line in text.splitlines():
        if line.startswith("•"):
            break
        head.append(line)
    return " ".join(head) or text


def _decision_labels(chart: Flowchart, node: FlowNode) -> list[str]:
    return [e.label for e in chart.out_edges(node.id)]


def extract_decision_answers(
    question: str, chart: Flowchart, cfg: DictConfig
) -> dict[str, str]:
    """Respuestas a las decisiones del arbol que el texto del usuario permite
    dar explicitamente. Las demas se omiten (se preguntaran si hacen falta)."""
    decisions = decision_overview(chart)
    catalog = "\n".join(
        f"- {d.id}: {_headline(d.text)} [{' | '.join(_decision_labels(chart, d))}]"
        for d in decisions
    )
    client = Client(host=cfg.env.ollama_url, timeout=llm_timeout(cfg))
    response = client.chat(
        model=cfg.agents.coordinator_model,
        messages=[
            {
                "role": "system",
                "content": _EXTRACT_SYSTEM_PROMPT_TEMPLATE.format(
                    catalog=catalog, unknown=_LLM_UNKNOWN
                ),
            },
            {"role": "user", "content": question},
        ],
        format=decisions_schema(
            {d.id: _decision_labels(chart, d) for d in decisions}, _LLM_UNKNOWN
        ),
        options=ollama_options(cfg, 800),
        stream=False,
    )
    content = response["message"]["content"].strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        logger.warning(
            "Agente arbol: extraccion inicial no es JSON valido (%r)", content
        )
        return {}
    if not isinstance(parsed, dict):
        return {}
    answers = {}
    for decision in decisions:
        value = parsed.get(decision.id)
        if isinstance(value, str) and value in _decision_labels(chart, decision):
            answers[decision.id] = value
    return answers


def parse_choice(raw_text: str, options: list[str]) -> str | None:
    """Interpreta la respuesta libre del usuario a una decision: una de las
    `options`, UNKNOWN si dice que no sabe, o None si no se pudo interpretar."""
    text = normalize(str(raw_text or "")).strip(" .,;:!¡?¿")
    if not text:
        return None
    if _UNKNOWN_RE.match(text) and not text.startswith("no se ha"):
        return UNKNOWN
    if text.isdigit() and 1 <= int(text) <= len(options):
        return options[int(text) - 1]
    for option in options:
        if normalize(option) == text:
            return option
    negative = [o for o in options if normalize(o).startswith("no")]
    positive = [o for o in options if o not in negative]
    if text in _YES_WORDS and len(positive) == 1:
        return positive[0]
    if text in _NO_WORDS and len(negative) == 1:
        return negative[0]
    words = set(re.findall(r"[a-z0-9]+", text))
    matches = [
        o for o in options if set(re.findall(r"[a-z0-9]+", normalize(o))) <= words
    ]
    return matches[0] if len(matches) == 1 else None


def _phrase_question(chart: Flowchart, node: FlowNode, options: list[str]) -> str:
    parts = [f"**{node.text}**"]
    for support in chart.support_nodes(node.id):
        parts.append(support.text)
    if len(options) == 2 and {normalize(o).split()[0] for o in options} == {"si", "no"}:
        choices = " / ".join(options)
    else:
        choices = "\n".join(f"{i}. {o}" for i, o in enumerate(options, 1))
    parts.append(
        f"Responde con una opción:\n{choices}\n\nSi no tienes el dato, responde «no sé»."
    )
    return "\n\n".join(parts)


# Presentacion del resultado ------------------------------------------------


def _citation_key(node: FlowNode) -> str:
    if node.section:
        return node.section
    if node.source_ref.startswith("p."):
        return f"protocolo {node.source_ref}"
    return node.source_ref or node.id


def _citations(chart: Flowchart, nodes: list[FlowNode]) -> list[dict[str, Any]]:
    """Una cita por seccion del diagrama (o pagina del protocolo indicada en
    el), con el texto literal de los nodos usados."""
    grouped: dict[str, list[str]] = {}
    for node in nodes:
        texts = grouped.setdefault(_citation_key(node), [])
        if node.text not in texts:
            texts.append(node.text)
    return [
        {
            "source_file": chart.source_file,
            "section": key,
            "page": 1,
            "verbatim_text": "\n".join(texts),
        }
        for key, texts in grouped.items()
    ]


def _indent(text: str) -> str:
    return text.replace("\n", "\n  ")


def _render_walk(chart: Flowchart, result: WalkResult) -> dict[str, Any]:
    steps: list[Step] = [s for s in result.steps if s.node.kind != "start"]
    lines = [
        f"Recorrido del árbol clínico {chart.protocol_id} {chart.version}:",
        "",
    ]
    for step in steps:
        if step.choice is not None:
            lines.append(f"- {_headline(step.node.text)} → **{step.choice}**")
        else:
            lines.append(f"- {_indent(step.node.text)}")
    if result.notes:
        lines += ["", "Además, según el diagrama:"]
        lines += [f"- {label}: {_indent(node.text)}" for label, node in result.notes]
    cited = [s.node for s in steps] + [node for _, node in result.notes]
    terminal = steps[-1].node.id if steps else chart.start_id
    return {
        "protocol_id": chart.protocol_id,
        "protocol_version": chart.version,
        "rule_id": terminal,
        "mode": "caso",
        "path": [s.node.id for s in steps],
        "citations": _citations(chart, cited),
        "answer_text": "\n".join(lines),
    }


def _when(conditions: list[tuple[FlowNode, str]]) -> str:
    return " y ".join(f"«{_headline(d.text)}» = {label}" for d, label in conditions)


def _consult_item(chart: Flowchart, node: FlowNode) -> str:
    """Una rama del arbol como enunciado autocontenido: su condicion completa
    y el texto del nodo. Antes las ramas hermanas iban anidadas bajo el
    "Aplica si" de otro nodo y el integrador llego a mezclar sus condiciones
    (p. ej. aplico a la gestante la condicion "sin procedimiento")."""
    conditions = chart.conditions_to(node.id)
    prefix = f"Cuando {_when(conditions)}: " if conditions else ""
    return f"- {prefix}{_indent(node.text)}"


def _render_consult(chart: Flowchart, node_ids: list[str]) -> dict[str, Any]:
    """Nodos relacionados con la pregunta y, para cada uno, las otras ramas de
    la ultima decision que lo condiciona; cada rama con su condicion completa."""
    lines = [
        f"Ramas del árbol clínico {chart.protocol_id} {chart.version} relacionadas con la consulta:"
    ]
    cited: list[FlowNode] = []
    shown: set[str] = set()

    def add(node: FlowNode) -> None:
        shown.add(node.id)
        cited.append(node)
        lines.extend(["", _consult_item(chart, node)])

    for node_id in node_ids:
        if node_id in shown:
            continue
        add(chart.nodes[node_id])
        conditions = chart.conditions_to(node_id)
        if conditions:
            last_decision, _taken = conditions[-1]
            for edge in chart.out_edges(last_decision.id):
                if edge.target not in shown:
                    add(chart.nodes[edge.target])
    return {
        "protocol_id": chart.protocol_id,
        "protocol_version": chart.version,
        "rule_id": node_ids[0],
        "rule_ids": list(node_ids),
        "mode": "consulta",
        "citations": _citations(chart, cited),
        "answer_text": "\n".join(lines),
    }


def _render_overview(chart: Flowchart) -> dict[str, Any]:
    """Para preguntas generales: la primera parte del arbol, en el orden del
    recorrido (antes se listaban todas las decisiones sin contexto, y el
    verificador lo juzgaba no relacionado)."""
    section, nodes = initial_section(chart)
    where = f" («{section}»)" if section else ""
    lines = [
        f"Inicio del árbol clínico {chart.protocol_id} {chart.version}{where}, en el orden del recorrido:",
        "",
    ]
    for node in nodes:
        if node.is_decision:
            lines.append(f"- {_headline(node.text)} → {' / '.join(_decision_labels(chart, node))}")
        else:
            lines.append(f"- {_indent(node.text)}")
    return {
        "protocol_id": chart.protocol_id,
        "protocol_version": chart.version,
        "rule_id": "inicio_del_arbol",
        "mode": "consulta",
        "citations": _citations(chart, nodes),
        "answer_text": "\n".join(lines),
    }


_SELECT_NODES_PROMPT_TEMPLATE = """\
Eres un asistente que ubica, en el arbol de decision de un protocolo clinico, \
los nodos que responden una pregunta. Recibes la lista de nodos (id, seccion \
del arbol y primera linea de su texto).

Elige hasta {max_nodes} nodos que respondan DIRECTAMENTE la pregunta, del mas \
al menos pertinente. Prefiere criterios, decisiones y tratamientos sobre \
pasos intermedios. Si ninguno la responde, devuelve una lista vacia. No \
respondas la pregunta.

Nodos:
{catalog}

Responde UNICAMENTE con un objeto JSON {{"nodos": ["<id>", ...]}}.
"""

_MAX_SELECTED_NODES = 4


def select_nodes(chart: Flowchart, question: str, cfg: DictConfig) -> list[str]:
    """Nodos que responden una pregunta informativa, elegidos por el modelo
    local entre los nodos del arbol (salida restringida a ids existentes).

    La coincidencia de palabras o de embeddings fallaba con preguntas
    conceptuales: para "¿que factores considerar al decidir el manejo?"
    devolvia "¿Tomografia compatible con neumonia?" en vez de las
    indicaciones de hospitalizacion.
    """
    nodes = [n for n in chart.nodes.values() if n.kind != "start"]
    catalog = "\n".join(
        f"- {n.id} [{n.section or n.source_ref or n.kind}]: {_headline(n.text)[:140]}" for n in nodes
    )
    client = Client(host=cfg.env.ollama_url, timeout=llm_timeout(cfg))
    response = client.chat(
        model=cfg.agents.coordinator_model,
        messages=[
            {
                "role": "system",
                "content": _SELECT_NODES_PROMPT_TEMPLATE.format(catalog=catalog, max_nodes=_MAX_SELECTED_NODES),
            },
            {"role": "user", "content": question},
        ],
        format=node_selection_schema([n.id for n in nodes], _MAX_SELECTED_NODES),
        options=ollama_options(cfg, 120),
        stream=False,
    )
    parsed = json.loads(response["message"]["content"])
    selected = [i for i in parsed.get("nodos", []) if i in chart.nodes]
    return list(dict.fromkeys(selected))[:_MAX_SELECTED_NODES]


def _consult_ranking(chart: Flowchart, question: str, cfg: DictConfig) -> tuple[list[str], bool]:
    """Nodos para el modo consulta y si los eligio el modelo: los que elige el
    modelo local; si la llamada falla, coincidencia de palabras (BM25)."""
    try:
        selected = select_nodes(chart, question, cfg)
        if selected:
            return selected, True
    except Exception:  # un fallo del modelo no debe dejar al arbol sin respuesta
        logger.exception("Agente arbol: fallo la seleccion de nodos; se usa BM25.")
    return rank_nodes_for_query(chart, question), False


# Nodos del grafo ------------------------------------------------------------


def tree_prepare_node(state: AgentState, cfg: DictConfig) -> dict:
    """Decide protocolo y modo; en modo caso, responde lo que el texto permita."""
    protocol_id, mode = classify_question(state["question"], cfg)
    if protocol_id is None:
        logger.info(
            "Agente arbol: ningun protocolo institucional aplica a esta pregunta."
        )
        return {"tree_protocol_id": None, "tree_initial_values": {}, "tree_mode": mode}

    if mode == "consulta":
        logger.info("Agente arbol: protocolo=%s modo=consulta", protocol_id)
        return {
            "tree_protocol_id": protocol_id,
            "tree_initial_values": {},
            "tree_mode": mode,
        }

    chart = get_flowchart(protocol_id)
    answers = extract_decision_answers(state["question"], chart, cfg)
    logger.info(
        "Agente arbol: protocolo=%s modo=caso respuestas_iniciales=%s",
        protocol_id,
        answers,
    )
    return {
        "tree_protocol_id": protocol_id,
        "tree_initial_values": answers,
        "tree_mode": mode,
    }


def tree_agent_node(state: AgentState, cfg: DictConfig) -> dict:
    """Nodo con interrupt(): recorre el arbol y, si hace falta, pregunta."""
    protocol_id = state.get("tree_protocol_id")
    if not protocol_id:
        return {
            "tree_result": None,
            "tree_questions_asked": 0,
            "tree_outcome": None,
            "tree_stop_detail": None,
        }

    chart = get_flowchart(protocol_id)

    if state.get("tree_mode") == "consulta":
        node_ids, by_model = _consult_ranking(chart, state["question"], cfg)
        result = (
            _render_consult(chart, node_ids) if node_ids else _render_overview(chart)
        )
        # Los nodos elegidos por el modelo a partir de la pregunta ya pasaron
        # un juicio de relevancia (verifier.py no repite el de relacion).
        result["nodes_selected_by_model"] = bool(node_ids) and by_model
        logger.info(
            "Agente arbol: modo consulta nodos=%s protocolo=%s",
            node_ids or "resumen",
            protocol_id,
        )
        return {
            "tree_result": result,
            "tree_questions_asked": 0,
            "tree_outcome": None,
            "tree_stop_detail": None,
        }

    answers: dict[str, str] = dict(state.get("tree_initial_values") or {})
    asked = 0
    while True:
        result = walk(chart, answers)
        if result.status != "pendiente" or asked >= MAX_QUESTIONS:
            break
        node = result.stopped_at
        assert node is not None
        raw_answer = interrupt(
            {
                "type": "tree_question",
                "protocol_id": protocol_id,
                "node_id": node.id,
                "question": _phrase_question(chart, node, result.options),
            }
        )
        asked += 1
        choice = parse_choice(str(raw_answer), result.options)
        if choice is not None:
            answers[node.id] = choice

    if result.status == "completo":
        rendered = _render_walk(chart, result)
        logger.info(
            "Agente arbol: recorrido completo hasta %s (protocolo=%s)",
            rendered["rule_id"],
            protocol_id,
        )
        return {
            "tree_result": rendered,
            "tree_questions_asked": asked,
            "tree_outcome": None,
            "tree_stop_detail": None,
        }

    stopped = _headline(result.stopped_at.text) if result.stopped_at else ""
    if result.status == "sin_ruta":
        outcome = "ausencia_de_coincidencias"
        detail = (
            f"el caso no cumple ninguna de las opciones del diagrama en «{stopped}»."
        )
    else:
        outcome = "informacion_faltante"
        detail = f"falta el dato para responder «{stopped}»."
    logger.info(
        "Agente arbol: recorrido detenido (%s) en %s tras %d preguntas",
        result.status,
        result.stopped_at and result.stopped_at.id,
        asked,
    )
    return {
        "tree_result": None,
        "tree_questions_asked": asked,
        "tree_outcome": outcome,
        "tree_stop_detail": detail,
    }
