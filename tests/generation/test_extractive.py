"""Modo extractivo del RAG (generation/extractive.py)."""

from __future__ import annotations

import json

from gpc_rag.common.models import Chunk, RetrievedChunk
from gpc_rag.generation.extractive import (
    REFUSAL,
    TRANSLATED_MARK,
    build_units,
    extractive_answer,
    segment,
)

_INDICATIONS = RetrievedChunk(
    chunk=Chunk(
        text="6. TRATAMIENTO — Indicaciones de tratamiento de bacteriuria asintomática\n\n"
        "- Mujer embarazada\n\n- Procedimiento invasivo de vías urinarias con riesgo de sangrado",
        source_file="CT-PL-197.pdf",
        section="6. TRATAMIENTO — Indicaciones de tratamiento de bacteriuria asintomática",
        page=3,
        chunk_index=17,
        chunk_id="a",
    ),
    score=0.5,
)
_ROW = RetrievedChunk(
    chunk=Chunk(
        text="Tabla 2. Gestante — Bacteriuria asintomática.\nCondición: Bacteriuria asintomática\n"
        "Tratamiento (1): Nitrofurantoína, Cefalexina, Fosfomicina (3)\n"
        "Dosis: Nitrofurantoína 100 mg cada 6 h\n¿Requiere urocultivo de control?: No",
        source_file="CT-PL-197.pdf",
        section="Tabla 2. Gestante — Bacteriuria asintomática",
        page=4,
        chunk_index=30,
        chunk_id="b",
    ),
    score=0.9,
)
_IDSA = RetrievedChunk(
    chunk=Chunk(
        text="Introduction\n\nTreatment of asymptomatic bacteriuria does not prevent UTI and "
        "may predispose to recurrent UTI. Complicated UTI is a common reason for admission.",
        source_file="idsa.pdf",
        section="Introduction",
        page=3,
        chunk_id="c",
        source_type="guia",
        source_title="IDSA 2025",
    ),
    score=0.8,
)


def test_lists_keep_their_lead_and_table_rows_stay_whole() -> None:
    assert segment(_INDICATIONS) == [
        (
            "Indicaciones de tratamiento de bacteriuria asintomática: Mujer embarazada; "
            "Procedimiento invasivo de vías urinarias con riesgo de sangrado."
        )
    ]
    (row,) = segment(_ROW)
    assert row.startswith("Tabla 2. Gestante — Bacteriuria asintomática. Condición:")
    assert row.endswith("¿Requiere urocultivo de control?: No.")
    assert row.count("Tabla 2. Gestante") == 1


class _FakeGenerator:
    def __init__(self, ids: list[str]):
        self.ids = ids
        self.prompts: list[str] = []

    def chat_json(self, system, user, schema, num_predict):
        self.prompts.append(user)
        if "traducciones" in schema["properties"]:
            uids = schema["properties"]["traducciones"]["required"]
            return json.dumps(
                {"traducciones": {u: "Tratar la bacteriuria asintomática no previene la ITU." for u in uids}}
            )
        allowed = set(schema["properties"]["ids"]["items"]["enum"])
        # Devuelve los ids de esta seleccion (protocolo o externas) y uno
        # inexistente, que debe ignorarse.
        return json.dumps({"ids": [*(i for i in self.ids if i in allowed), "u999"]})


def test_answer_is_literal_units_with_citations_protocol_first_in_document_order() -> None:
    sources = [_ROW, _IDSA, _INDICATIONS]  # orden de recuperacion por puntaje
    units = {u.text[:20]: u.uid for u in build_units(sources)}
    idsa_uid = next(uid for text, uid in units.items() if text.startswith("Treatment"))
    row_uid = next(uid for text, uid in units.items() if text.startswith("Tabla 2"))
    ind_uid = next(uid for text, uid in units.items() if text.startswith("Indicaciones"))
    answer = extractive_answer(_FakeGenerator([idsa_uid, row_uid, ind_uid]), "¿Cuándo tratar?", sources)
    lines = answer.splitlines()
    assert lines[0].startswith("- Indicaciones de tratamiento") and "Pagina: 3]" in lines[0]
    assert lines[1].startswith("- Tabla 2.") and "Pagina: 4]" in lines[1]
    assert lines[2].startswith("- Tratar la bacteriuria asintomática no previene la ITU. " + TRANSLATED_MARK)
    assert "[Guia: idsa.pdf, Seccion: Introduction, Pagina: 3]" in lines[2]


def test_no_selected_units_is_an_abstention() -> None:
    assert extractive_answer(_FakeGenerator([]), "¿Malaria?", [_IDSA]) == REFUSAL


def test_captions_alone_are_not_units_and_ids_are_removed_from_translations() -> None:
    from gpc_rag.generation.extractive import translate_units

    rc = RetrievedChunk(
        chunk=Chunk(
            text="Asymptomatic Bacteriuria\n\nBox 2. Main Indications to Treat Asymptomatic "
            "Bacteriuria\n\nThe first is pregnant women, because treatment lowers the risk of "
            "pyelonephritis.",
            source_file="ajkd.pdf",
            section="Asymptomatic Bacteriuria",
            chunk_id="d",
            source_type="revision",
        ),
        score=1.0,
    )
    units = build_units([rc])
    assert [u.text[:9] for u in units] == ["The first"]

    class _Gen:
        def chat_json(self, system, user, schema, num_predict):
            return json.dumps({"traducciones": {"u1": "u1: La primera son las gestantes."}})

    assert translate_units(_Gen(), units) == {"u1": "La primera son las gestantes."}


def test_numeric_callouts_and_inline_bullets_are_cleaned() -> None:
    rc = RetrievedChunk(
        chunk=Chunk(
            text="Box\n\nMain indications: pregnancy; • urologic procedures with mucosal bleeding [6].",
            source_file="x.pdf",
            section="Box",
            chunk_id="e",
        ),
        score=1.0,
    )
    assert segment(rc) == ["Main indications: pregnancy; urologic procedures with mucosal bleeding."]


def test_sentences_that_refer_to_the_previous_one_keep_it() -> None:
    rc = RetrievedChunk(
        chunk=Chunk(
            text="Riesgo\n\nLa colonizacion previa por Pseudomonas aumenta el riesgo de infeccion. "
            "No contamos con datos locales para validar este hallazgo en nuestra poblacion.",
            source_file="x.pdf",
            section="Riesgo",
            chunk_id="f",
        ),
        score=1.0,
    )
    (unit,) = segment(rc)
    assert unit.startswith("La colonizacion previa") and unit.endswith("en nuestra poblacion.")


def test_quiz_answers_are_removed() -> None:
    rc = RetrievedChunk(
        chunk=Chunk(
            text="Review\n\nPregnancy is an indication for the treatment of ASB, so the answer is (a).",
            source_file="ajkd.pdf",
            section="Review",
            chunk_id="g",
        ),
        score=1.0,
    )
    assert segment(rc) == ["Pregnancy is an indication for the treatment of ASB."]


def test_long_lists_are_split_and_reference_numbers_removed() -> None:
    items = "\n".join(
        f"- Item number {i} about screening asymptomatic bacteriuria in adults with a long "
        f"explanation of the reasons and the evidence behind it 1 52 53" for i in range(6)
    )
    rc = RetrievedChunk(
        chunk=Chunk(
            text=f"Screening tests\n\nScreening tests:\n{items}\n\nAdults older than 65.",
            source_file="ck.pdf",
            section="Screening tests",
            chunk_id="h",
            source_type="revision",
        ),
        score=1.0,
    )
    units = [u.text for u in build_units([rc])]
    assert len(units) == 7
    assert units[0] == (
        "Screening tests: Item number 0 about screening asymptomatic bacteriuria in adults "
        "with a long explanation of the reasons and the evidence behind it."
    )
    assert units[-1] == "Adults older than 65."  # un numero solo no se toca


def test_low_relevance_abstains_without_calling_the_selector() -> None:
    low = RetrievedChunk(chunk=_IDSA.chunk, score=0.02)
    gen = _FakeGenerator(["u1"])
    assert extractive_answer(gen, "¿Malaria?", [low], min_relevance=0.1) == REFUSAL
    assert gen.prompts == []


def test_definitions_are_not_a_table_row_and_wrapped_items_are_joined() -> None:
    rc = RetrievedChunk(
        chunk=Chunk(
            text="3. DEFINICIONES\n\n- Bacteriuria asintomática: recuento significativo sin "
            "síntomas atribuibles a ITU, con dos urocultivos en mujeres no gestantes y uno en "
            "hombres y gestantes, tomados de forma apropiada en el servicio de urgencias.\n"
            "- ITU complicada: ITU que ocurre en presencia de alteraciones estructurales o\n"
            "- funcionales de las vías urinarias.\n"
            "- Cistitis: infección aguda de la vejiga con bacteriuria significativa y sin fiebre.",
            source_file="itu.pdf",
            section="3. DEFINICIONES",
            chunk_id="i",
        ),
        score=1.0,
    )
    (unit,) = segment(rc)  # lista corta: una unidad, no una "fila de tabla"
    assert unit.startswith("3. DEFINICIONES: Bacteriuria asintomática:")
    assert "estructurales o funcionales de las vías urinarias; Cistitis:" in unit


def test_figure_footnotes_are_split_and_dot_references_removed() -> None:
    rc = RetrievedChunk(
        chunk=Chunk(
            text="Figure 2\n\nRisk factors are previous MRSA infection and high local MRSA "
            "prevalence.8 †Advice from the 2025 ATS guideline. ‡Warning signs are continued fever.",
            source_file="lancet.pdf",
            section="Figure 2",
            chunk_id="j",
            source_type="revision",
        ),
        score=1.0,
    )
    assert [u.text for u in build_units([rc])] == [
        "Risk factors are previous MRSA infection and high local MRSA prevalence.",
        "Advice from the 2025 ATS guideline.",
        "Warning signs are continued fever.",
    ]


def test_minimum_final_score_drops_low_fragments() -> None:
    from gpc_rag.retrieval.hybrid import with_institutional_quota

    low = RetrievedChunk(chunk=_IDSA.chunk, score=-0.1)
    high = RetrievedChunk(chunk=_ROW.chunk, score=0.6)
    assert with_institutional_quota([high, low], 8, 3, 0.05, min_final_score=0.05) == [high]


def test_protocol_and_external_units_are_selected_separately() -> None:
    sources = [_ROW, _IDSA, _INDICATIONS]
    gen = _FakeGenerator([])
    extractive_answer(gen, "¿Cuándo tratar?", sources)
    assert len(gen.prompts) == 2
    assert "Tabla 2" in gen.prompts[0] and "Treatment of asymptomatic" not in gen.prompts[0]
    assert "Treatment of asymptomatic" in gen.prompts[1] and "Tabla 2" not in gen.prompts[1]


def test_bibliographic_references_are_not_units() -> None:
    review = RetrievedChunk(
        chunk=Chunk(
            text="Asymptomatic Bacteriuria\n\nScreening is not recommended in nonpregnant adults. "
            "Nicolle LE, Gupta K, Bradley SF, et al. Clinical practice guideline for the "
            "management of asymptomatic bacteriuria. Clin Infect Dis. 2019;68(10). "
            "https://doi.org/10.1093/cid/ciy1121 +ESSENTIAL READING",
            source_file="ITU review 2024.pdf",
            section="Asymptomatic Bacteriuria",
            page=6,
            chunk_id="r",
            source_type="revision",
        ),
        score=0.4,
    )
    texts = [u.text for u in build_units([review])]
    assert any("not recommended" in t for t in texts)
    assert not any("et al" in t or "doi" in t for t in texts)


def test_study_summaries_from_external_guides_are_not_units() -> None:
    guide = RetrievedChunk(
        chunk=Chunk(
            text="14. Factores de riesgo para SAMR o P. aeruginosa\n\n"
            "Recomendamos cubrir P. aeruginosa en pacientes con aislamiento previo.\n\n"
            "El estudio de Prina introdujo el acrónimo PES con una mortalidad mayor "
            "(OR 2.51, IC95%: 1.20-5.25).\n\n"
            "El análisis de Metersky y colaboradores mostró que la EPOC discriminaba riesgo de "
            "_Pseudomonas_ , mientras que la diabetes se asoció con SAMR_157 .",
            source_file="GPC NAC 2026.pdf",
            section="14. Factores de riesgo para SAMR o P. aeruginosa",
            page=18,
            chunk_id="g",
            source_type="guia",
        ),
        score=0.4,
    )
    texts = [u.text for u in build_units([guide])]
    assert texts == ["Recomendamos cubrir P. aeruginosa en pacientes con aislamiento previo."]


def test_cut_fragments_and_abbreviation_legends_are_not_units() -> None:
    review = RetrievedChunk(
        chunk=Chunk(
            text="Clinical features\n\nNo single clinical sign has good predictive\n\n"
            "Figure 1: Considerations for diagnosis ATS=American Thoracic Society. "
            "FiO2=fraction of inspired oxygen. IDSA=Infectious Diseases Society of America.\n\n"
            "Chest radiography is recommended to confirm the diagnosis in adults.",
            source_file="lancet.pdf",
            section="Clinical features",
            page=5,
            chunk_id="l",
            source_type="revision",
        ),
        score=0.4,
    )
    texts = [u.text for u in build_units([review])]
    assert texts == ["Chest radiography is recommended to confirm the diagnosis in adults."]
