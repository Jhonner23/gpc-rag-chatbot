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
from gpc_rag.ingestion.tables import strip_markdown_tables

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


# Pie de pagina institucional de los protocolos de la CUB. Se repite en cada
# pagina y, al indexarlo, generaba fragmentos que solo contenian este texto
# (y que el modelo llego a copiar en sus respuestas).
_BOILERPLATE_RES = [
    re.compile(
        r"_?El contenido de este documento es de propiedad y de uso exclusivo de la "
        r"Cl[ií]nica Universitaria Bolivariana\.?_?\.?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\**_?Cualquier impresi[oó]n o copia tomada de este documento se considera como "
        r"COPIA NO CONTROLADA_?\**",
        re.IGNORECASE,
    ),
    re.compile(r"\**\s*P[aá]gina\s+\d+\s+de\s+\d+\s*\**", re.IGNORECASE),
]
_EMPTY_MARKUP_LINE_RE = re.compile(r"^[\s*_\-]*$")

# Un "encabezado" de mas de esta longitud es en realidad un parrafo que el
# extractor marco como titulo (paso con el diagnostico del protocolo de ITU:
# todo el parrafo quedo como seccion y el fragmento solo tenia el pie de
# pagina). Se devuelve al texto.
_MAX_SECTION_TITLE_CHARS = 120


def clean_page_text(text: str) -> str:
    """Quita el pie de pagina institucional y las lineas que quedan vacias."""
    for pattern in _BOILERPLATE_RES:
        text = pattern.sub("", text)
    lines = [line.rstrip() for line in text.splitlines()]
    kept = [line if not _EMPTY_MARKUP_LINE_RE.match(line) else "" for line in lines]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


_CONTINUATION = "\x00continuacion"
_BIBLIOGRAPHY_RE = re.compile(r"referencias|bibliograf", re.IGNORECASE)


def _is_bibliography(title: str) -> bool:
    """La bibliografia no es contenido clinico, y sus titulos ("community-
    acquired pneumonia...") se colaban en la recuperacion por coincidencia
    de palabras."""
    return bool(_BIBLIOGRAPHY_RE.search(_strip_markdown_markers(title)))


def _with_section_header(title: str, text: str) -> str:
    """Antepone el titulo de la seccion al texto del fragmento.

    La busqueda (densa, BM25 y reranker) solo ve el texto: sin el titulo, un
    fragmento como la lista "- Mujer embarazada - Procedimiento invasivo..."
    de la seccion "Indicaciones de tratamiento de bacteriuria asintomatica"
    no contiene las palabras de la pregunta y nunca se recupera.
    """
    clean = _strip_markdown_markers(title)
    if not clean or clean == "Sin seccion" or _strip_markdown_markers(text).startswith(clean):
        return text
    return f"{clean}\n\n{text}"


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
        if len(title) > _MAX_SECTION_TITLE_CHARS:
            head = title.split(":", 1)[0].strip()
            body = f"{title}\n\n{body}".strip()
            title = head if 0 < len(head) <= _MAX_SECTION_TITLE_CHARS // 2 else default_title
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
    drop_tables_on_pages: set[int] | None = None,
) -> list[Chunk]:
    """Convierte las paginas extraidas de un PDF en una lista de Chunk listos para indexar.

    Args:
        pages: salida de `extract.extract_pages_markdown` -- lista de {"text", "page"}.
        source_file: nombre/identificador de la guia de origen (para citar la fuente).
        drop_tables_on_pages: paginas cuyas tablas extraidas se descartan porque
            se indexan desde el archivo de tablas revisadas (ingestion/tables.py).
    """
    chunks: list[Chunk] = []
    chunk_idx = 0
    drop_tables_on_pages = drop_tables_on_pages or set()
    # Si una pagina termina en la bibliografia, el texto sin encabezado al
    # inicio de la siguiente es su continuacion. No se hereda el titulo en
    # general: los titulos en negrita dentro del texto ("PLAN DE EGRESO") no
    # se detectan como encabezado, y heredar el anterior etiquetaba mal esos
    # fragmentos.
    previous_was_bibliography = False

    for page in pages:
        text = clean_page_text(page["text"])
        if page["page"] in drop_tables_on_pages:
            text = strip_markdown_tables(text)
        if not text:
            continue
        sections = split_into_sections(text, default_title=_CONTINUATION)
        continues_bibliography = previous_was_bibliography
        previous_was_bibliography = bool(sections) and (
            _is_bibliography(sections[-1].title)
            or (sections[-1].title == _CONTINUATION and continues_bibliography)
        )
        for section in sections:
            if section.title == _CONTINUATION:
                if continues_bibliography:
                    continue
                section.title = "Sin seccion"
            if _is_bibliography(section.title):
                continue
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
                        text=_with_section_header(section.title, piece),
                        source_file=source_file,
                        section=_strip_markdown_markers(section.title),
                        page=page["page"],
                        chunk_index=chunk_idx,
                        chunk_id=str(uuid.uuid4()),
                    )
                )
                chunk_idx += 1

    return chunks
