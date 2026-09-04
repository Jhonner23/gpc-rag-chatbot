"""Extraccion de texto de PDFs de Guias de Practica Clinica (GPC).

Usa pymupdf4llm (preserva encabezados/estructura como markdown, mejor que
extraer texto plano) y cae a OCR (ocrmypdf + Tesseract, idioma "spa") cuando
una pagina no tiene capa de texto -- es decir, cuando es una imagen escaneada.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import fitz  # PyMuPDF
import pymupdf4llm

logger = logging.getLogger(__name__)

MIN_CHARS_PER_PAGE_TO_SKIP_OCR = 20


def page_has_text_layer(pdf_path: Path, page_index: int) -> bool:
    """True si la pagina ya tiene texto seleccionable (no necesita OCR)."""
    with fitz.open(pdf_path) as doc:
        page = doc[page_index]
        return len(page.get_text().strip()) >= MIN_CHARS_PER_PAGE_TO_SKIP_OCR


def needs_ocr(pdf_path: Path) -> bool:
    """True si el PDF tiene al menos una pagina sin capa de texto (escaneada)."""
    with fitz.open(pdf_path) as doc:
        return any(
            len(doc[i].get_text().strip()) < MIN_CHARS_PER_PAGE_TO_SKIP_OCR
            for i in range(len(doc))
        )


def ocr_pdf(pdf_path: Path, lang: str = "spa") -> Path:
    """Corre ocrmypdf sobre el PDF y devuelve la ruta del PDF con capa de texto.

    Requiere que el binario `ocrmypdf` (y Tesseract con el idioma `spa`) esten
    instalados en el sistema -- no es una dependencia de Python.
    """
    out_path = Path(tempfile.gettempdir()) / f"ocr_{pdf_path.stem}.pdf"
    cmd = [
        "ocrmypdf",
        "--skip-text",  # no reprocesa paginas que ya tienen texto
        "--language",
        lang,
        str(pdf_path),
        str(out_path),
    ]
    logger.info("Corriendo OCR sobre %s ...", pdf_path.name)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603 -- cmd es una lista fija, sin input de shell
    if result.returncode != 0:
        raise RuntimeError(
            f"ocrmypdf fallo para {pdf_path.name}: {result.stderr[-2000:]}"
        )
    return out_path


def extract_pages_markdown(pdf_path: Path) -> list[dict]:
    """Extrae el PDF a markdown, pagina por pagina, preservando estructura.

    Aplica OCR automaticamente si detecta paginas escaneadas.

    Returns:
        Lista de dicts: {"text": str (markdown), "page": int (1-indexado)}.
    """
    pdf_path = Path(pdf_path)
    working_path = pdf_path
    if needs_ocr(pdf_path):
        logger.warning(
            "%s parece tener paginas escaneadas -- aplicando OCR.", pdf_path.name
        )
        working_path = ocr_pdf(pdf_path)

    pages = pymupdf4llm.to_markdown(str(working_path), page_chunks=True)
    return [
        {
            "text": page["text"],
            "page": page.get("metadata", {}).get("page", idx + 1),
        }
        for idx, page in enumerate(pages)
        if page["text"].strip()
    ]
