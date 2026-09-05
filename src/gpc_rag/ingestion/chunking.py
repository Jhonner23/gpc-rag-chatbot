"""Chunking jerarquico por seccion para documentos clinicos.

No se usa un splitter de tamano fijo "ciego": primero se corta por encabezados
markdown (secciones reales de la GPC) y solo si una seccion sigue siendo muy
larga se subdivide, sin romper nunca una recomendacion a la mitad de una frase.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from gpc_rag.common.models import Chunk

_HEADER_RE = re.compile(r"^(#{1,4})\s+(.*)$", re.MULTILINE)

# Paginas que pasaron por OCR (ver ingestion/extract.py) no traen encabezados
# markdown reales (#, ##...) -- el texto queda "plano". Muchas GPC sin embargo
# marcan sus preguntas clinicas numeradas en negrita al inicio de linea, ej.
# "**13. En pacientes adultos con diagnostico de NAC grave...**". Sin detectar
# esto como limite de seccion, toda la pagina cae en una sola seccion "Sin
# seccion" larga, que el splitter por parrafos corta a ciegas -- pudiendo
# separar una recomendacion clinica (con el farmaco/dosis exacta) de su propia
# pregunta y contexto, y dejar que el LLM reciba solo la mitad.
_BOLD_NUMBERED_HEADER_RE = re.compile(r"^\*\*(\d{1,3}\.\s.*)$", re.MULTILINE)


def _strip_markdown_markers(text: str) -> str:
    """Quita ** y _ de enfasis para dejar un titulo de seccion legible."""
    return re.sub(r"[*_]+", "", text).strip()


def approx_token_count(text: str) -> int:
    """Aproximacion rapida de tokens sin depender de un tokenizer real.

    ~0.75 palabras por token es una aproximacion razonable para espanol con
    modelos tipo Llama/Qwen. Suficiente para dimensionar chunks; no se usa
    para facturacion ni limites duros del modelo.
    """
    words = len(text.split())
    return max(1, int(words / 0.75))


@dataclass
class _Section:
    title: str
    text: str


def split_into_sections(markdown_text: str, default_title: str = "Sin seccion") -> list[_Section]:
    """Divide un texto markdown en secciones usando encabezados reales (#, ##...)
    y, como respaldo (paginas OCR sin markdown), preguntas numeradas en negrita
    al inicio de linea (ej. "**13. En pacientes...**").
    """
    raw_matches: list[tuple[int, int, str]] = [
        (m.start(), m.end(), m.group(2).strip()) for m in _HEADER_RE.finditer(markdown_text)
    ]
    raw_matches += [
        (m.start(), m.end(), _strip_markdown_markers(m.group(1)))
        for m in _BOLD_NUMBERED_HEADER_RE.finditer(markdown_text)
    ]
    matches = sorted(raw_matches, key=lambda t: t[0])

    if not matches:
        return [_Section(title=default_title, text=markdown_text.strip())]

    sections: list[_Section] = []
    if matches[0][0] > 0:
        preamble = markdown_text[: matches[0][0]].strip()
        if preamble:
            sections.append(_Section(title=default_title, text=preamble))

    for i, (_start, end, title) in enumerate(matches):
        body_end = matches[i + 1][0] if i + 1 < len(matches) else len(markdown_text)
        body = markdown_text[end:body_end].strip()
        if body:
            sections.append(_Section(title=title, text=body))

    return sections


def _split_long_section(
    text: str, chunk_size_tokens: int, overlap_ratio: float, min_chunk_tokens: int
) -> list[str]:
    """Subdivide una seccion larga en pedazos con solape, cortando por parrafo/frase."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = approx_token_count(para)
        if current and current_tokens + para_tokens > chunk_size_tokens:
            chunks.append("\n\n".join(current))
            overlap_tokens = int(chunk_size_tokens * overlap_ratio)
            kept: list[str] = []
            kept_tokens = 0
            for p in reversed(current):
                t = approx_token_count(p)
                if kept_tokens + t > overlap_tokens:
                    break
                kept.insert(0, p)
                kept_tokens += t
            current = kept
            current_tokens = kept_tokens
        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))

    return [c for c in chunks if approx_token_count(c) >= min_chunk_tokens] or [text]


def chunk_document(
    pages: list[dict],
    source_file: str,
    chunk_size_tokens: int = 400,
    chunk_overlap_ratio: float = 0.15,
    min_chunk_tokens: int = 40,
) -> list[Chunk]:
    """Convierte las paginas extraidas de un PDF en una lista de Chunk listos para indexar.

    Args:
        pages: salida de `extract.extract_pages_markdown` -- lista de {"text", "page"}.
        source_file: nombre/identificador de la guia de origen (para citar la fuente).
    """
    chunks: list[Chunk] = []
    chunk_idx = 0

    for page in pages:
        sections = split_into_sections(page["text"])
        for section in sections:
            if approx_token_count(section.text) <= chunk_size_tokens:
                pieces = [section.text]
            else:
                pieces = _split_long_section(
                    section.text, chunk_size_tokens, chunk_overlap_ratio, min_chunk_tokens
                )
            for piece in pieces:
                if approx_token_count(piece) < min_chunk_tokens:
                    continue
                chunks.append(
                    Chunk(
                        text=piece,
                        source_file=source_file,
                        section=section.title,
                        page=page["page"],
                        chunk_index=chunk_idx,
                        chunk_id=str(uuid.uuid4()),
                    )
                )
                chunk_idx += 1

    return chunks
