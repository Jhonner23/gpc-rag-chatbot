"""Controles deterministas de fundamentacion (common/grounding.py), con
textos tomados de respuestas reales que los jueces locales aprobaron."""

from __future__ import annotations

from gpc_rag.common.grounding import (
    grounding_problems,
    mentions_other_guides,
    unknown_citations,
)

_FILE = "CT-PL-197_Protocolo_ITU_Adultos_v6-sep-2025.pdf"
_TREE = "Árbol clínico CT-PL-197 v06"


def test_citation_to_retrieved_file_is_accepted() -> None:
    text = f"No se trata [Guia: {_FILE}, Seccion: 3. DEFINICIONES, Pagina: 1]."
    assert unknown_citations(text, [_FILE]) == []
    assert (
        unknown_citations(
            "[Guía: CT-PL-197_Protocolo_ITU_Adultos_v6-sep-2025, Pagina: 4]", [_FILE]
        )
        == []
    )


def test_invented_document_citation_is_detected() -> None:
    text = (
        "[Guía: Protocolo de Infección del Tracto Urinario en Adultos (CT-PL-197), "
        "Sección: Bacteriuria Asintomática, Página: No especificada]"
    )
    assert unknown_citations(text, [_FILE, _TREE]) == [
        "Protocolo de Infección del Tracto Urinario en Adultos (CT-PL-197)"
    ]


def test_other_guides_with_a_single_document_is_detected() -> None:
    text = "La guía presenta tres opciones, mientras que otras guías pueden ofrecer más alternativas."
    assert mentions_other_guides(text, [_FILE, _FILE])
    assert grounding_problems(text, [_FILE])


def test_negated_or_multi_document_mentions_are_allowed() -> None:
    assert not mentions_other_guides(
        "No hay otras guías en el corpus sobre este tema.", [_FILE]
    )
    assert not mentions_other_guides(
        "Algunas guías difieren.", [_FILE, "CT-PL-193_Protocolo_NAC_v9.pdf"]
    )
    assert grounding_problems("Texto sin citas ni comparaciones.", [_FILE]) == []


def test_agreement_between_guides_with_a_single_document_is_detected() -> None:
    text = "Todas las fuentes coinciden en que no se requiere urocultivo de control."
    assert mentions_other_guides(text, [_FILE])
    assert mentions_other_guides("Ambas guías recomiendan nitrofurantoína.", [_FILE])
    assert not mentions_other_guides(
        "Todas las fuentes recuperadas provienen del mismo protocolo.", [_FILE]
    )
