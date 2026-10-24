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
    assert (
        _admin_penalty("1. ¿Radiografía de tórax? — Necesidades de investigación futura", "x", 0.3)
        == -0.3
    )
    assert _admin_penalty("2. ALCANCE", "x", 0.0) == 0.0
    assert _admin_penalty("Resumen", "manejo inicial", 0.3) == -0.3
    assert _admin_penalty("Methods", "x", 0.3, source_type="guia", page=5) == -0.3
    assert _admin_penalty("Treatment", "x", 0.3, source_type="guia", page=1) == -0.3
    assert _admin_penalty("Treatment", "x", 0.3, source_type="guia", page=4) == 0.0
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


def test_admin_sections_are_excluded_from_the_context_when_configured() -> None:
    from omegaconf import OmegaConf

    from gpc_rag.common.models import Chunk, RetrievedChunk
    from gpc_rag.retrieval.hybrid import HybridRetriever

    class _Reranker:
        def predict(self, pairs):
            return [0.9 for _ in pairs]

    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.cfg = OmegaConf.create(
        {"retrieval": {"exclude_admin_sections": True, "admin_section_penalty": 0.3}}
    )
    retriever._reranker = _Reranker()
    chunks = [
        RetrievedChunk(chunk=Chunk(text="t", source_file="g.pdf", section=s, chunk_id=s, page=7), score=0)
        for s in ("Pregunta 1 — Necesidades de investigación futura", "Tratamiento empírico")
    ]
    ranked = retriever._rerank("¿Tratamiento de la NAC?", chunks, 8)
    assert [r.chunk.section for r in ranked] == ["Tratamiento empírico"]


def test_question_outside_the_corpus_is_detected() -> None:
    from omegaconf import OmegaConf

    from gpc_rag.retrieval.hybrid import HybridRetriever

    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.cfg = OmegaConf.create({"retrieval": {"topic_terms": ["neumonia", "nac", "pielonefritis"]}})
    retriever._vocabulary = frozenset(
        {"antibiotico", "recomendado", "tratamiento", "pacientes", "adultos", "neumonia", "influenza"}
    )
    malaria = "¿Cuál es el tratamiento recomendado para la malaria en adultos?"
    assert retriever.out_of_corpus_terms(malaria) == ["malaria"]
    # Con un tema del corpus se responde lo que haya del tema.
    assert retriever.out_of_corpus_terms("¿Tratamiento de la pielonefritis con lupus?") == []
    assert retriever.out_of_corpus_terms("¿Tratamiento de la influenza en adultos?") == []


def test_verbs_and_short_topic_terms_do_not_trigger_the_out_of_corpus_guard() -> None:
    from omegaconf import OmegaConf

    from gpc_rag.retrieval.hybrid import HybridRetriever

    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.cfg = OmegaConf.create({"retrieval": {"topic_terms": ["itu", "nac"]}})
    retriever._vocabulary = frozenset({"pseudomonas", "aeruginosa", "tratamiento", "empirico"})
    assert retriever.out_of_corpus_terms("¿Cuándo cubrir Pseudomonas aeruginosa?") == []
    assert retriever.out_of_corpus_terms("¿Criterios de egreso en ITU para pasar a vía oral?") == []


def test_external_documents_are_capped_but_the_protocol_is_not() -> None:
    from gpc_rag.common.models import Chunk, RetrievedChunk
    from gpc_rag.retrieval.hybrid import with_institutional_quota

    def rc(i, source, kind, score):
        chunk = Chunk(text="t", source_file=source, chunk_id=f"{source}{i}", source_type=kind)
        return RetrievedChunk(chunk=chunk, score=score)

    ranked = [rc(i, "lancet.pdf", "revision", 0.9 - i / 100) for i in range(5)]
    ranked += [rc(i, "cub.pdf", "institucional", 0.5 - i / 100) for i in range(4)]
    ranked += [rc(0, "gpc.pdf", "guia", 0.3)]
    chosen = with_institutional_quota(ranked, 8, 3, 0.05, max_per_external_document=3)
    files = [r.chunk.source_file for r in chosen]
    assert files.count("lancet.pdf") == 3
    assert files.count("cub.pdf") == 4
    assert "gpc.pdf" in files  # sin el tope, los 2 fragmentos extra del Lancet lo dejaban fuera


def test_guide_front_matter_is_administrative() -> None:
    from gpc_rag.retrieval.hybrid import _admin_penalty

    query = "¿Cuál es el manejo inicial de la NAC?"
    assert _admin_penalty("Población de interés", query, 0.3, "x", "guia", 4) < 0
    assert _admin_penalty("Usuarios de la guía", query, 0.3, "x", "guia", 4) < 0
    assert _admin_penalty("Población de interés", "¿Cuál es la población de interés?", 0.3) == 0


def _topic_retriever(tmp_path):
    from collections import Counter

    from omegaconf import OmegaConf

    from gpc_rag.retrieval.hybrid import HybridRetriever

    (tmp_path / "fuentes.yaml").write_text(
        "documentos:\n"
        "  - archivo: GPC NAC 2026.pdf\n    titulo: GPC NAC\n    tipo: guia\n    tema: NAC\n"
        "  - archivo: ITU review.pdf\n    titulo: ITU review\n    tipo: revision\n    tema: ITU\n",
        encoding="utf-8",
    )
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.cfg = OmegaConf.create(
        {
            "retrieval": {
                "topics": {"NAC": ["neumonia", "nac"], "ITU": ["urinaria", "itu", "bacteriuria"]},
                "institutional_topics": {"CT-PL-193": "NAC", "CT-PL-197": "ITU"},
                "external_catalog_dir": str(tmp_path),
                "rare_term_ratio": 0.01,
            }
        }
    )
    retriever._external_topics = None
    retriever._bm25_chunks = [object()] * 500
    retriever._doc_freq = Counter({"meningitis": 1, "bacteriana": 40, "adultos": 300, "esputo": 4})
    retriever._vocabulary = frozenset(retriever._doc_freq)
    return retriever


def test_chunks_of_another_topic_are_dropped(tmp_path) -> None:
    from gpc_rag.common.models import Chunk, RetrievedChunk

    retriever = _topic_retriever(tmp_path)

    def rc(source, kind):
        return RetrievedChunk(chunk=Chunk(text="t", source_file=source, source_type=kind), score=0.5)

    pool = [
        rc("CT-PL-197_Protocolo_ITU.pdf", "institucional"),
        rc("CT-PL-193_Protocolo_NAC.pdf", "institucional"),
        rc("GPC NAC 2026.pdf", "guia"),
        rc("ITU review.pdf", "revision"),
        rc("sin catalogo.pdf", "revision"),
    ]
    kept = retriever.same_topic("¿En qué casos de infección urinaria se piden hemocultivos?", pool)
    assert [r.chunk.source_file for r in kept] == [
        "CT-PL-197_Protocolo_ITU.pdf",
        "ITU review.pdf",
        "sin catalogo.pdf",
    ]
    # Sin tema o con dos temas no se filtra.
    assert retriever.same_topic("¿Cuándo es de buena calidad el esputo?", pool) == pool
    assert retriever.same_topic("¿Neumonía o infección urinaria?", pool) == pool


def test_answer_that_does_not_mention_the_asked_disease_is_uncovered(tmp_path) -> None:
    retriever = _topic_retriever(tmp_path)
    question = "¿Cómo se maneja la meningitis bacteriana en adultos?"
    assert retriever.uncovered_terms(question, "Recomendamos ceftriaxona en adultos con NAC.") == [
        "meningitis"
    ]
    esputo = "¿Cuándo se considera de buena calidad una muestra de esputo?"
    assert retriever.uncovered_terms(esputo, "Muestra de esputo con <10 células epiteliales.") == []
    # Con un tema del corpus no se exige.
    assert retriever.uncovered_terms("¿Meningitis en paciente con neumonía?", "Ceftriaxona.") == []


def test_admin_section_is_named_by_its_stem() -> None:
    from gpc_rag.retrieval.hybrid import _admin_penalty

    query = "¿Cómo se clasifican la fuerza de la recomendación y la calidad de la evidencia?"
    assert _admin_penalty("CRITERIO DE CLASIFICACIÓN DE LAS RECOMENDACIONES", query, 0.3) == 0.0
