"""Catalogo de fuentes externas (ingestion/sources.py) y limpieza de
articulos de revista (ingestion/chunking.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpc_rag.ingestion.chunking import chunk_document
from gpc_rag.ingestion.sources import load_catalog, source_for

_EXTERNAS = Path("data/01_raw/externas")


def test_catalog_covers_the_clinic_documents_and_tolerates_zip_names() -> None:
    catalog = load_catalog(_EXTERNAS)
    assert len(catalog) == 12
    # unzip deja "elecci#U00f3n" en lugar de "elección".
    info = source_for(Path("Guia ITU- IDSA elecci#U00f3n AB 2025.pdf"), catalog)
    assert info.source_type == "guia" and info.topic == "ITU"
    assert source_for(Path("JAMA NAC review.pdf"), catalog).source_type == "revision"
    with pytest.raises(ValueError, match="no esta en fuentes.yaml"):
        source_for(Path("otro.pdf"), catalog)


def _journal_pages(n: int) -> list[dict]:
    body = (
        "## Question {i}\n\n"
        "Patients without risk factors should receive amoxicillin 1 g three times "
        "daily for {i} days according to the panel recommendation.\n\n"
    )
    return [
        {
            "text": "### <mark>AMERICAN THORACIC SOCIETY DOCUMENTS</mark>\n\n"
            + body.format(i=i + 1)
            + "Downloaded from jamanetwork.com by Universidad user on 08/26/2026",
            "page": i + 1,
        }
        for i in range(n)
    ]


def test_running_headers_and_footers_are_removed() -> None:
    chunks = chunk_document(_journal_pages(6), source_file="x.pdf", min_chunk_tokens=5)
    assert chunks
    assert all("AMERICAN THORACIC" not in c.text for c in chunks)
    assert all("Downloaded from" not in c.text for c in chunks)
    assert [c.section for c in chunks] == [f"Question {i}" for i in range(1, 7)]


def test_references_are_not_indexed_even_without_a_heading() -> None:
    refs = "\n".join(
        f"**{i}.** Smith J, Doe A, et al. Some trial. _Chest_ . 2021;159(1):58-72. "
        f"doi:10.1016/j.chest.{i}"
        for i in range(10, 16)
    )
    pages = [
        {
            "text": "## Treatment\n\nUse ceftriaxone 1 g IV daily in hospitalized adults "
            "with community-acquired pneumonia.\n\n## ARTICLE INFORMATION\n\nAuthor "
            "contributions listed.\n\n" + refs,
            "page": 1,
        }
    ]
    chunks = chunk_document(pages, source_file="x.pdf", min_chunk_tokens=5)
    assert [c.section for c in chunks] == ["Treatment"]
