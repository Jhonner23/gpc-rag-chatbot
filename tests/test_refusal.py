"""Deteccion de rechazos (common/refusal.py)."""

from __future__ import annotations

from gpc_rag.common.refusal import is_refusal


def test_short_refusal_is_detected() -> None:
    assert is_refusal(
        "No encuentro informacion suficiente en las guias disponibles para responder "
        "esto con seguridad. Las guias proporcionadas se centran en ITU y neumonia, "
        "pero no abordan el tratamiento de la malaria en adultos."
    )


def test_long_answer_ending_in_a_refusal_is_not_an_abstention() -> None:
    """Caso real: deduce un tratamiento y al final dice que no hay dosis. No
    puede saltarse el evaluador como si fuera una abstencion."""
    answer = (
        "El enfoque recomendado para el manejo inicial de la neumonia no se detalla "
        "en el contexto. No obstante, se puede inferir que se recomienda un abordaje "
        "empirico basado en la etiologia mas frecuente y el riesgo de germenes "
        "resistentes. El agente causal mas frecuente es Streptococcus pneumoniae, "
        "seguido de Haemophilus influenzae y Staphylococcus aureus. En resumen, el "
        "enfoque inicial probablemente incluiria un antibiotico empirico que cubra "
        "Streptococcus pneumoniae y Staphylococcus aureus. No encuentro informacion "
        "suficiente para proporcionar una dosis especifica."
    )
    assert not is_refusal(answer)
