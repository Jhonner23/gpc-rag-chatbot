"""Limpieza del texto extraido y tablas revisadas (ingestion/chunking.py,
ingestion/tables.py). Los textos de prueba vienen de la extraccion real de
los protocolos CT-PL-193 y CT-PL-197."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpc_rag.ingestion.chunking import (
    chunk_document,
    clean_page_text,
    split_into_sections,
)
from gpc_rag.ingestion.tables import (
    load_curated_tables,
    row_text,
    strip_markdown_tables,
    table_chunks,
)

_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "01_raw" / "gpc"
_ITU_PDF = "CT-PL-197_Protocolo_ITU_Adultos_v6-sep-2025.pdf"
_NAC_PDF = "CT-PL-193_Protocolo_NAC_v9-ago-2024.pdf"


def test_institutional_footer_is_removed() -> None:
    text = (
        "Texto clinico.\n\n"
        "_El contenido de este documento es de propiedad y de uso exclusivo de la Clínica Universitaria "
        "Bolivariana._ **_Cualquier impresión o copia tomada de este documento se considera como "
        "COPIA NO CONTROLADA_ Página 5 de 5**\n\n"
        "**Página 1 de 7**"
    )
    assert clean_page_text(text) == "Texto clinico."


def test_footer_only_page_produces_no_chunks() -> None:
    footer = (
        "_El contenido de este documento es de propiedad y de uso exclusivo de la Clínica Universitaria "
        "Bolivariana._\n\n**_Cualquier impresión o copia tomada de este documento se considera como "
        "COPIA NO CONTROLADA_ Página 4 de 5**"
    )
    assert chunk_document([{"text": footer, "page": 4}], source_file="x.pdf") == []


def test_paragraph_marked_as_heading_goes_back_to_the_text() -> None:
    title = (
        "5. Diagnóstico: Análisis de orina: "
        + "Se debe realizar citoquímico de orina. " * 5
    )
    sections = split_into_sections(f"## {title}\nResto del texto.")
    assert sections[0].title == "5. Diagnóstico"
    assert "Se debe realizar citoquímico de orina" in sections[0].text


def test_markdown_tables_are_stripped() -> None:
    text = "Antes.\n\n|A|B|\n|---|---|\n|1|2|\n\nDespues."
    assert strip_markdown_tables(text) == "Antes.\n\nDespues."


def _copy(tmp_path: Path, pdf_name: str, *, reviewed: bool) -> Path:
    source = _RAW_DIR / pdf_name.replace(".pdf", ".tablas.yaml")
    text = source.read_text(encoding="utf-8")
    if not reviewed:
        text = text.replace("revisado: true", "revisado: false")
    (tmp_path / source.name).write_text(text, encoding="utf-8")
    return tmp_path / pdf_name


def _reviewed_copy(tmp_path: Path, pdf_name: str) -> Path:
    return _copy(tmp_path, pdf_name, reviewed=True)


def test_unreviewed_tables_are_not_indexed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no esta marcado como revisado"):
        load_curated_tables(_copy(tmp_path, _ITU_PDF, reviewed=False))


def test_versioned_tables_are_reviewed() -> None:
    for pdf in (_ITU_PDF, _NAC_PDF):
        assert load_curated_tables(_RAW_DIR / pdf) is not None


def test_asymptomatic_bacteriuria_row_carries_its_note(tmp_path: Path) -> None:
    curated = load_curated_tables(_reviewed_copy(tmp_path, _ITU_PDF))
    table_1 = curated.tables[0]
    text = row_text(table_1, table_1.rows[1])
    assert text.startswith(
        "Tabla 1. Tratamiento de la infección del tracto urinario (ITU) en adultos "
        "en la CUB — Bacteriuria asintomática."
    )
    assert "CONDICIÓN: Bacteriuria asintomática**" in text
    assert "Recordar su tratamiento sólo en gestantes" in text
    assert curated.drop_extracted_tables_on_pages == {4, 5}


def test_nac_treatment_table_is_indexed(tmp_path: Path) -> None:
    curated = load_curated_tables(_reviewed_copy(tmp_path, _NAC_PDF))
    chunks = table_chunks(curated, start_index=10)
    assert len(chunks) == 5
    assert chunks[0].chunk_index == 10
    assert all(c.page == 5 and c.source_file == _NAC_PDF for c in chunks)
    assert any(
        "Piperacilina tazobactam 4.5 g IV cada 8 horas" in c.text for c in chunks
    )
    assert all("Oseltamivir 75 mg VO" in c.text for c in chunks)


def test_row_section_names_the_table_and_the_condition(tmp_path: Path) -> None:
    curated = load_curated_tables(_reviewed_copy(tmp_path, _ITU_PDF))
    sections = {c.section for c in table_chunks(curated)}
    assert (
        "Tabla 2. Consideraciones manejo de la ITU en la paciente gestante — "
        "Bacteriuria asintomática"
    ) in sections
