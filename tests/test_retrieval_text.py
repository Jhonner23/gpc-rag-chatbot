"""Texto que ven la busqueda lexica (BM25) y el reranker: tokenizador y
titulo de seccion en cada fragmento. Casos tomados del diagnostico real con
la pregunta de bacteriuria asintomatica."""

from __future__ import annotations

from gpc_rag.ingestion.chunking import chunk_document
from gpc_rag.retrieval.hybrid import _recommendation_boost, _tokenize


def test_tokenizer_ignores_punctuation_accents_and_stopwords() -> None:
    assert _tokenize("Bacteriuria asintomática**") == ["bacteriuria", "asintomatica"]
    assert _tokenize("¿En qué situaciones una bacteriuria asintomática debe ser tratada y en cuáles no?") == [
        "situaciones",
        "bacteriuria",
        "asintomatica",
        "tratada",
        "no",
    ]


def test_numbered_section_boost_is_disabled_by_default() -> None:
    assert _recommendation_boost("1. OBJETIVO", 0.25) == 0.0
    assert _recommendation_boost("1. OBJETIVO", 0.25, boost=0.3) == 0.3


def test_section_title_is_part_of_the_chunk_text() -> None:
    page = (
        "**Indicaciones de tratamiento de bacteriuria asintomática**\n\n"
        + "- Mujer embarazada\n\n- Procedimiento invasivo de vías urinarias con riesgo de sangrado "
        + "o disrupción del uroepitelio\n\n" * 3
    ).replace("**Indicaciones", "## **Indicaciones")
    chunks = chunk_document([{"text": page, "page": 3}], source_file="x.pdf", min_chunk_tokens=5)
    assert chunks[0].section == "Indicaciones de tratamiento de bacteriuria asintomática"
    assert chunks[0].text.startswith("Indicaciones de tratamiento de bacteriuria asintomática\n\n- Mujer embarazada")
