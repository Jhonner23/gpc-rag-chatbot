"""Utilidades puras sobre el motor de `cpg_tree` (sin LLM, sin I/O).

Separado de `tree_agent.py` para poder probarlo de forma aislada: dado un
`EvaluationResult` (salida de `cpg_tree.engine.evaluate_package`) y el
`ProtocolVersion` del que salio, determina que variables, si se conocieran,
tienen mas probabilidad de destrabar una conclusion (outcome MATCHED) --
sin eso, el wizard tendria que preguntar por las 60-76 variables del
protocolo en vez de priorizar.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from cpg_tree.engine import Case
from cpg_tree.engine.rules import EvaluationResult, RuleOutcome
from cpg_tree.knowledge.conditions import Condition, LogicalExpression, LogicalOperand
from cpg_tree.knowledge.enums import ValidationStatus
from cpg_tree.knowledge.protocol import ProtocolVersion
from cpg_tree.knowledge.rules import Rule


def collect_variable_refs(operand: LogicalOperand) -> set[str]:
    """Recorre un Condition/LogicalExpression y junta los variable_ref que usa."""
    if isinstance(operand, Condition):
        return {operand.variable_ref}
    if isinstance(operand, LogicalExpression):
        refs: set[str] = set()
        for child in operand.operands:
            refs |= collect_variable_refs(child)
        return refs
    raise TypeError(f"operando desconocido: {type(operand).__name__}")


def rule_variable_refs(rule: Rule) -> set[str]:
    """Todas las variables que una regla puede llegar a consultar (condicion,
    applies_to, excepciones)."""
    refs = collect_variable_refs(rule.condition)
    if rule.applies_to is not None:
        refs |= collect_variable_refs(rule.applies_to)
    for exception in rule.exceptions:
        refs |= collect_variable_refs(exception)
    return refs


def is_unknown_in_case(variable_id: str, case: Case) -> bool:
    """True si la variable no esta en el Case, o esta pero en estado UNKNOWN."""
    entry = case.values.get(variable_id)
    return entry is None or entry.value is None


def rank_blocking_variables(
    evaluation: EvaluationResult,
    package: ProtocolVersion,
    case: Case,
    *,
    only_validated: bool = True,
) -> list[tuple[str, int]]:
    """Ordena las variables aun desconocidas por cuantas reglas INDETERMINATE
    destrabarian si se conocieran.

    Solo se cuentan reglas con `validation_status == EXTRACTED` cuando
    `only_validated=True` -- nunca se le da peso a una regla UNRESOLVED para
    decidir que preguntarle al usuario, siguiendo la misma regla de
    auditoria que se usa para aceptar un resultado MATCHED.

    Devuelve una lista de (variable_id, cantidad_de_reglas_bloqueadas),
    ordenada de mayor a menor impacto.
    """
    counter: Counter[str] = Counter()
    for rule_eval in evaluation.rule_results:
        if rule_eval.outcome is not RuleOutcome.INDETERMINATE:
            continue
        if only_validated and rule_eval.validation_status is not ValidationStatus.EXTRACTED:
            continue
        rule = package.rules[rule_eval.rule_id]
        for variable_id in rule_variable_refs(rule):
            if is_unknown_in_case(variable_id, case):
                counter[variable_id] += 1
    return counter.most_common()


def has_validated_match(
    evaluation: EvaluationResult,
    rule_results_by_outcome: Mapping[str, RuleOutcome] | None = None,
) -> bool:
    """True si al menos una regla EXTRACTED (nunca UNRESOLVED) dio MATCHED."""
    del rule_results_by_outcome  # reservado para uso futuro (debug/trazabilidad)
    return any(
        r.outcome is RuleOutcome.MATCHED and r.validation_status is ValidationStatus.EXTRACTED
        for r in evaluation.rule_results
    )


def matched_validated_rules(evaluation: EvaluationResult):
    """Reglas MATCHED y EXTRACTED de un EvaluationResult (auditadas)."""
    return [
        r
        for r in evaluation.rule_results
        if r.outcome is RuleOutcome.MATCHED and r.validation_status is ValidationStatus.EXTRACTED
    ]
