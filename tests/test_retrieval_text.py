"""Texto que ven la busqueda lexica (BM25) y el reranker: tokenizador y
titulo de seccion en cada fragmento. Casos tomados del diagnostico real con
la pregunta de bacteriuria asintomatica."""

from __future__ import annotations

from gpc_rag.ingestion.chunking import chunk_document
from gpc_rag.retrieval.hybrid import (
    _admin_penalty,
    _bm25_query_tokens,
    _recommendation_boost,
    _tokenize,
    retrieval_query,
)


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


def test_bibliography_and_its_continuation_are_not_indexed() -> None:
    refs = "- Lim WS, et al. BTS guidelines for community acquired pneumonia. Thorax. 2009. " * 3
    pages = [
        {"text": "## Plan de egreso\n\nTexto clinico del egreso. " * 5 + "\n\n## REFERENCIAS BIBLIOGRÁFICAS:\n\n" + refs, "page": 6},
        {"text": refs + "\n\n## Control\n\nTexto que si se indexa. " * 3, "page": 7},
    ]
    chunks = chunk_document(pages, source_file="x.pdf", min_chunk_tokens=5)
    assert all("BTS guidelines" not in c.text for c in chunks)
    assert {c.section for c in chunks} == {"Plan de egreso", "Control"}


def test_retrieval_query_drops_subquestions_about_the_corpus() -> None:
    question = (
        "¿Qué recomendaciones presentan las guías disponibles sobre el manejo de la "
        "bacteriuria asintomática? ¿Qué diferencias existen entre las recomendaciones "
        "encontradas para el manejo de esta condición? ¿Qué fuentes del corpus contienen "
        "información relacionada con esta recomendación?"
    )
    assert retrieval_query(question) == (
        "¿Qué recomendaciones presentan las guías disponibles sobre el manejo de la "
        "bacteriuria asintomática?"
    )
    assert _bm25_query_tokens(retrieval_query(question)) == [
        "manejo",
        "bacteriuria",
        "asintomatica",
    ]
    # Preguntas simples o con subpreguntas clinicas no cambian.
    simple = "¿En qué situaciones una bacteriuria asintomática debe ser tratada?"
    assert retrieval_query(simple) == simple
    assert retrieval_query("¿Cuál es la dosis? ¿Y en gestantes?") == (
        "¿Cuál es la dosis? ¿Y en gestantes?"
    )


def test_admin_sections_are_penalized_unless_the_question_names_them() -> None:
    assert _admin_penalty("2. ALCANCE", "¿Qué factores considerar en neumonía?", 0.3) == -0.3
    assert _admin_penalty("4. INTRODUCCIÓN — Epidemiología", "manejo", 0.3) == -0.3
    assert _admin_penalty("CONTROL DE CAMBIOS — 06", "manejo", 0.3) == -0.3
    assert _admin_penalty("2. ALCANCE", "¿Cuál es el alcance del protocolo?", 0.3) == 0.0
    # Subsecciones clinicas dentro de la introduccion y definiciones: sin cambio.
    assert _admin_penalty("4. INTRODUCCIÓN — Manifestaciones clínicas", "x", 0.3) == 0.0
    assert _admin_penalty("3. DEFINICIONES", "x", 0.3) == 0.0
    assert _admin_penalty("2. ALCANCE", "x", 0.0) == 0.0
    header = "Proceso: Cuidado y tratamiento Código: CT-PL-193 Versión: 9"
    assert _admin_penalty("PROTOCOLO DE NAC", "manejo inicial", 0.3, header) == -0.3
    assert _admin_penalty("PROTOCOLO DE NAC", "¿Qué versión está vigente?", 0.3, header) == 0.0


def test_institutional_quota_reserves_relevant_protocol_fragments() -> None:
    from gpc_rag.common.models import Chunk, RetrievedChunk
    from gpc_rag.retrieval.hybrid import with_institutional_quota

    def rc(i: int, score: float, kind: str) -> RetrievedChunk:
        return RetrievedChunk(
            chunk=Chunk(text=str(i), source_file=f"{kind}.pdf", chunk_id=str(i), source_type=kind),
            score=score,
        )

    ranked = [rc(i, 0.9 - i * 0.05, "guia") for i in range(8)] + [
        rc(10, 0.30, "institucional"),
        rc(11, 0.20, "institucional"),
        rc(12, 0.01, "institucional"),  # irrelevante: no se fuerza
    ]
    chosen = with_institutional_quota(ranked, top_k=8, quota=3, min_score=0.05)
    ids = [r.chunk.chunk_id for r in chosen]
    assert len(chosen) == 8
    assert "10" in ids and "11" in ids and "12" not in ids
    assert [r.score for r in chosen] == sorted((r.score for r in chosen), reverse=True)
    # Sin cupo, solo puntaje.
    assert all(
        r.chunk.source_type == "guia"
        for r in with_institutional_quota(ranked, top_k=8, quota=0, min_score=0.05)
    )


def test_english_stopwords_are_ignored() -> None:
    assert _tokenize("The treatment of the pneumonia is not") == ["treatment", "pneumonia", "not"]
