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
        assert set(self.ids) <= set(schema["properties"]["ids"]["items"]["enum"])
        return json.dumps({"ids": [*self.ids, "u999"]})  # un id inexistente se ignora


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
