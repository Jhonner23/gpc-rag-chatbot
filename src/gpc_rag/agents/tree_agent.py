"""Agente arbol: evalua deterministicamente un Case contra el motor cpg_tree.

Flujo (dentro del grafo, ver agents/graph.py):

1. ``tree_prepare_node`` (nodo normal, corre una sola vez): decide que
   protocolo institucional aplica (NAC/ITU/ninguno) y hace una primera
   extraccion de variables desde la pregunta original del usuario, con un
   LLM. Si se repite el nodo `tree_agent` por un `interrupt()`, este nodo NO
   se vuelve a ejecutar (LangGraph solo reproduce el nodo que quedo pausado).

2. ``tree_agent_node`` (nodo con `interrupt()`): evalua el Case contra el
   motor determinista de `cpg_tree`; si ninguna regla EXTRACTED da MATCHED,
   pregunta -una variable a la vez, estilo wizard- por la que mas reglas
   indeterminadas este bloqueando, hasta encontrar una conclusion o agotar
   el limite de preguntas (MAX_QUESTIONS).

Solo se acepta como respuesta del arbol una regla con
`validation_status == EXTRACTED` (nunca UNRESOLVED): esa es la auditoria que
el usuario pidio explicitamente ("un agente que recorra el arbol y revise su
resultado").
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langgraph.types import interrupt
from ollama import Client
from omegaconf import DictConfig

from cpg_tree.engine import Case, evaluate_package
from cpg_tree.knowledge.enums import VariableType
from cpg_tree.knowledge.protocol import ProtocolVersion
from cpg_tree.knowledge.variables import Variable
from cpg_tree.protocols.itu_v06 import build_itu_package
from cpg_tree.protocols.nac_v09 import build_nac_package

from gpc_rag.agents.state import AgentState
from gpc_rag.agents.tree_support import matched_validated_rules, rank_blocking_variables

logger = logging.getLogger(__name__)

MAX_QUESTIONS = 8

_PACKAGE_BUILDERS = {
    "NAC": build_nac_package,
    "ITU": build_itu_package,
}

_PROTOCOL_SELECT_SYSTEM_PROMPT = """\
Eres un clasificador que decide a cual protocolo clinico institucional \
pertenece una pregunta, entre estos dos:

- NAC: Protocolo de neumonia adquirida en comunidad (diagnostico, \
severidad/CURB-65, criterios de ingreso a UCI, manejo ambulatorio vs \
hospitalario, alta).
- ITU: Protocolo de infeccion del tracto urinario y bacteriuria \
asintomatica en poblacion adulta (ITU alta/baja, pielonefritis, embarazo, \
cateter, tratamiento).

Si la pregunta no corresponde claramente a ninguno de los dos protocolos, \
responde NINGUNO.

Responde UNICAMENTE con un objeto JSON de esta forma exacta, sin texto \
adicional:
{"protocol": "NAC"} o {"protocol": "ITU"} o {"protocol": "NINGUNO"}
"""

_EXTRACT_SYSTEM_PROMPT_TEMPLATE = """\
Eres un extractor de datos clinicos. A partir de la pregunta de un usuario, \
identifica SOLO los valores de las siguientes variables que esten \
explicitamente mencionados o se puedan inferir con certeza del texto. NO \
inventes valores que no esten en la pregunta -- si no se menciona, no la \
incluyas.

Variables disponibles:
{catalog}

Responde UNICAMENTE con un objeto JSON plano {{"variable_id": valor, ...}} \
usando solo ids de la lista anterior. Usa true/false para variables \
booleanas, numeros (sin unidades ni texto) para variables numericas, y el \
texto exacto de uno de los valores permitidos para variables categoricas. \
Si no hay ningun dato identificable, responde {{}}.
"""

_SI_WORDS = {"si", "sí", "s", "yes", "y", "verdadero", "true", "positivo", "afirmativo"}
_NO_WORDS = {"no", "n", "false", "falso", "negativo"}


def select_protocol(question: str, cfg: DictConfig) -> str | None:
    """Devuelve "NAC", "ITU", o None si ningun protocolo institucional aplica."""
    client = Client(host=cfg.env.ollama_url)
    response = client.chat(
        model=cfg.agents.coordinator_model,
        messages=[
            {"role": "system", "content": _PROTOCOL_SELECT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        format="json",
        options={"temperature": 0.0, "num_predict": 50},
        stream=False,
    )
    content = response["message"]["content"].strip()
    try:
        parsed = json.loads(content)
        protocol = str(parsed.get("protocol", "")).strip().upper()
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Agente arbol: no se pudo parsear la seleccion de protocolo (%r)", content)
        return None
    if protocol in _PACKAGE_BUILDERS:
        return protocol
    return None


def _variable_catalog_text(variables: dict[str, Variable]) -> str:
    lines = []
    for var in sorted(variables.values(), key=lambda v: v.id):
        type_info = var.type.value
        if var.allowed_values:
            type_info += f" (valores permitidos: {', '.join(var.allowed_values)})"
        lines.append(f"- {var.id} [{type_info}]: {var.label}")
    return "\n".join(lines)


def _coerce_llm_value(raw_value: Any, variable: Variable) -> Any | None:
    """Convierte el valor crudo del LLM al tipo de la variable, o None si no cuadra."""
    if raw_value is None:
        return None
    if variable.type is VariableType.BOOLEAN:
        return raw_value if isinstance(raw_value, bool) else None
    if variable.type in (VariableType.NUMERIC, VariableType.DURATION):
        if isinstance(raw_value, bool):
            return None
        if isinstance(raw_value, (int, float)):
            return float(raw_value)
        return None
    if variable.type is VariableType.CATEGORICAL:
        if not isinstance(raw_value, str):
            return None
        if not variable.allowed_values:
            return raw_value
        for option in variable.allowed_values:
            if option.lower() == raw_value.lower():
                return option
        return None
    return None


def extract_case_values(
    question: str, package: ProtocolVersion, cfg: DictConfig
) -> dict[str, Any]:
    """Primera pasada: extrae lo que se pueda inferir de la pregunta original.

    Tolerante: una clave invalida o un valor con el tipo equivocado se
    descarta individualmente en vez de invalidar toda la extraccion.
    """
    client = Client(host=cfg.env.ollama_url)
    system_prompt = _EXTRACT_SYSTEM_PROMPT_TEMPLATE.format(
        catalog=_variable_catalog_text(package.variables)
    )
    response = client.chat(
        model=cfg.agents.coordinator_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        format="json",
        options={"temperature": 0.0, "num_predict": 500},
        stream=False,
    )
    content = response["message"]["content"].strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Agente arbol: extraccion inicial no es JSON valido (%r)", content)
        return {}
    if not isinstance(parsed, dict):
        return {}

    values: dict[str, Any] = {}
    for key, raw_value in parsed.items():
        variable = package.variables.get(key)
        if variable is None:
            continue
        coerced = _coerce_llm_value(raw_value, variable)
        if coerced is not None:
            values[key] = coerced
    return values


def _phrase_question(variable: Variable) -> str:
    if variable.type is VariableType.BOOLEAN:
        return f"{variable.label}: ¿si o no?"
    if variable.type is VariableType.CATEGORICAL and variable.allowed_values:
        opciones = ", ".join(variable.allowed_values)
        return f"{variable.label}: ¿cual de estas opciones? ({opciones})"
    if variable.type in (VariableType.NUMERIC, VariableType.DURATION):
        unidad = f" ({variable.unit})" if variable.unit else ""
        return f"{variable.label}{unidad}: ¿cual es el valor?"
    return f"{variable.label}:"


def _parse_answer(raw_text: str, variable: Variable) -> Any | None:
    """Interpreta la respuesta en texto libre del usuario para una variable.

    Devuelve None si no se pudo interpretar (el llamador decide si reintenta
    o descarta esa variable para no preguntar en bucle infinito).
    """
    text = (raw_text or "").strip().lower()
    if not text:
        return None
    if variable.type is VariableType.BOOLEAN:
        if text in _SI_WORDS:
            return True
        if text in _NO_WORDS:
            return False
        return None
    if variable.type in (VariableType.NUMERIC, VariableType.DURATION):
        match = re.search(r"-?\d+(?:[.,]\d+)?", text)
        if not match:
            return None
        return float(match.group(0).replace(",", "."))
    if variable.type is VariableType.CATEGORICAL:
        if not variable.allowed_values:
            return text
        for option in variable.allowed_values:
            if option.lower() == text or option.lower() in text:
                return option
        return None
    return None


def _render_tree_result(package: ProtocolVersion, rule_eval: Any) -> dict[str, Any]:
    rule = package.rules[rule_eval.rule_id]
    action_lines = []
    for action_ref in rule.action_refs:
        action = package.actions.get(action_ref)
        if action is not None:
            action_lines.append(action.label or action.id)

    citations = []
    for fragment_id in rule.provenance.fragment_refs:
        fragment = package.fragments.get(fragment_id)
        if fragment is None:
            continue
        document = package.documents.get(fragment.document_id) if fragment.document_id else None
        citations.append(
            {
                "source_file": document.filename if document else (fragment.document_id or "?"),
                "section": fragment.section or rule.id,
                "page": fragment.page,
                "verbatim_text": fragment.verbatim_text,
            }
        )

    answer_lines = [
        f"Segun el arbol de decision del protocolo {package.protocol.id} "
        f"({package.protocol.name}), regla '{rule.id}':",
    ]
    if action_lines:
        answer_lines.append("")
        answer_lines.extend(f"- {line}" for line in action_lines)

    return {
        "protocol_id": package.protocol.id,
        "rule_id": rule.id,
        "actions": action_lines,
        "citations": citations,
        "answer_text": "\n".join(answer_lines),
    }


def tree_prepare_node(state: AgentState, cfg: DictConfig) -> dict:
    """Nodo normal (sin interrupt): decide protocolo y hace la primera extraccion."""
    protocol_id = select_protocol(state["question"], cfg)
    if protocol_id is None:
        logger.info("Agente arbol: ningun protocolo institucional aplica a esta pregunta.")
        return {"tree_protocol_id": None, "tree_initial_values": {}}

    package = _PACKAGE_BUILDERS[protocol_id]()
    initial_values = extract_case_values(state["question"], package, cfg)
    logger.info("Agente arbol: protocolo=%s valores_iniciales=%s", protocol_id, initial_values)
    return {"tree_protocol_id": protocol_id, "tree_initial_values": initial_values}


def tree_agent_node(state: AgentState, cfg: DictConfig) -> dict:  # noqa: ARG001
    """Nodo con interrupt(): evalua y, si hace falta, pregunta estilo wizard."""
    protocol_id = state.get("tree_protocol_id")
    if not protocol_id:
        return {"tree_result": None, "tree_questions_asked": 0}

    package = _PACKAGE_BUILDERS[protocol_id]()
    case_values: dict[str, Any] = dict(state.get("tree_initial_values") or {})
    skipped: set[str] = set()
    questions_asked = 0
    matches: list[Any] = []

    for _ in range(MAX_QUESTIONS):
        case = Case.from_inputs(case_values)
        evaluation = evaluate_package(package, case)
        matches = matched_validated_rules(evaluation)
        if matches:
            break
        ranking = rank_blocking_variables(evaluation, package, case)
        candidate = next((var_id for var_id, _ in ranking if var_id not in skipped), None)
        if candidate is None:
            break
        variable = package.variables[candidate]
        raw_answer = interrupt(
            {
                "type": "tree_question",
                "protocol_id": protocol_id,
                "variable_id": candidate,
                "question": _phrase_question(variable),
            }
        )
        questions_asked += 1
        parsed = _parse_answer(str(raw_answer), variable)
        if parsed is None:
            skipped.add(candidate)
        else:
            case_values[candidate] = parsed
    else:
        case = Case.from_inputs(case_values)
        evaluation = evaluate_package(package, case)
        matches = matched_validated_rules(evaluation)

    if not matches:
        logger.info(
            "Agente arbol: sin conclusion tras %d preguntas (protocolo=%s)",
            questions_asked,
            protocol_id,
        )
        return {"tree_result": None, "tree_questions_asked": questions_asked}

    result = _render_tree_result(package, matches[0])
    logger.info("Agente arbol: MATCH regla=%s protocolo=%s", result["rule_id"], protocol_id)
    return {"tree_result": result, "tree_questions_asked": questions_asked}
