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

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

# Paginas que pasaron por OCR (ver ingestion/extract.py) no traen encabezados
# markdown reales (#, ##...) -- el texto queda "plano". Muchas GPC sin embargo
# marcan sus preguntas clinicas numeradas en negrita al inicio de linea, ej.
# "**13. En pacientes adultos con diagnostico de NAC grave...**". Sin detectar
# esto como limite de seccion, toda la pagina cae en una sola seccion "Sin
# seccion" larga, que el splitter por parrafos corta a ciegas -- pudiendo
# separar una recomendacion clinica (con el farmaco/dosis exacta) de su propia
# pregunta y contexto, y dejar que el LLM reciba solo la mitad.
_BOLD_NUMBERED_HEADER_RE = re.compile(r"^\*\*(\d{1,3}\.\s.*)$", re.MULTILINE)

# Los protocolos de la CUB marcan sus subsecciones con una etiqueta en
# negrita al inicio del parrafo, en la misma linea que el texto:
# "**DIAGNÓSTICO MICROBIOLÓGICO:** Los estudios...", "**Hemocultivos:**
# Deben tomarse...", "- **Indicaciones de hospitalización en pacientes con
# ITU:**". El extractor no las marca como encabezado, asi que quedaban en el
# fragmento de la seccion anterior: los criterios de hemocultivos de NAC se
# etiquetaban "ESTUDIOS IMAGINOLÓGICOS" y se mezclaban con los examenes para
# definir hospitalizacion, y la lista de indicaciones de hospitalizacion de
# ITU se leia como indicaciones de tratar la bacteriuria asintomatica.
_INLINE_LABEL_RE = re.compile(
    r"^(?:[-•]\s*)?\*\*(?P<label>[^*\n]{2,90}?)\s*(?P<inner>[:.])?\s*\*\*\s*(?P<outer>[:.])?",
    re.MULTILINE,
)
# Leyendas de graficos: no son subsecciones. Las de tabla si se conservan,
# porque las notas al pie de la tabla quedan debajo de su leyenda.
_CAPTION_RE = re.compile(r"^(gr[aá]fico|figura)\s*\d", re.IGNORECASE)
_TABLE_CAPTION_RE = re.compile(r"^\*\*(?P<label>Tabla\s*\d+\.[^*\n]*?)\s*\*\*\s*$", re.MULTILINE)

# Una subseccion con titulo propio se indexa aunque sea corta (p. ej. "Los
# estudios microbiologicos solo se haran en paciente con indicacion de
# hospitalizacion"); el minimo general de tokens aplica al texto sin titulo.
_MIN_TITLED_SECTION_TOKENS = 10


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
# Texto que el extractor saca por OCR de las imagenes (leyendas de graficos,
# cajas del flujograma). Llega desordenado ("—a. stoc eopecers: reese de
# riesgo para P. aeruginosa; MRSA ote") y el modelo llego a deducir
# recomendaciones a partir de el. El flujograma lo cubre el agente del arbol.
_PICTURE_TEXT_RE = re.compile(
    r"<!--\s*Start of picture text\s*-->.*?<!--\s*End of picture text\s*-->", re.DOTALL
)
# Leyenda del grafico cuyo texto OCR se quito: sola no tiene contenido
# ("Gráfico 1. Etiología de la neumonía... en la CUB") y quedaba primera en
# BM25 para cualquier pregunta sobre la neumonia adquirida en comunidad.
_PICTURE_CAPTION_RE = re.compile(r"^\*\*(gr[aá]fico|figura)\s*\d+\.?\*\*.*$", re.IGNORECASE | re.MULTILINE)
# Etiquetas HTML que el extractor deja en los articulos (<mark>, <u>, <sup>).
_HTML_TAG_RE = re.compile(r"</?(mark|u|sup|sub|span|br)\b[^>]*>", re.IGNORECASE)
_EMPTY_MARKUP_LINE_RE = re.compile(r"^[\s*_\-]*$")

# Un "encabezado" de mas de esta longitud es en realidad un parrafo que el
# extractor marco como titulo (paso con el diagnostico del protocolo de ITU:
# todo el parrafo quedo como seccion y el fragmento solo tenia el pie de
# pagina). Se devuelve al texto.
_MAX_SECTION_TITLE_CHARS = 120


def clean_page_text(text: str) -> str:
    """Quita el pie de pagina institucional, el texto OCR de imagenes y las
    lineas que quedan vacias."""
    text = _PICTURE_TEXT_RE.sub("", text)
    text = _PICTURE_CAPTION_RE.sub("", text)
    text = _HTML_TAG_RE.sub("", text)
    for pattern in _BOILERPLATE_RES:
        text = pattern.sub("", text)
    lines = [line.rstrip() for line in text.splitlines()]
    kept = [line if not _EMPTY_MARKUP_LINE_RE.match(line) else "" for line in lines]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


_CONTINUATION = "\x00continuacion"
_BIBLIOGRAPHY_RE = re.compile(
    r"referencias|bibliograf|references|additional readings?|lecturas recomendadas"
    # Secciones de los articulos que no son contenido clinico.
    r"|article information|author contributions|conflict of interest|disclosures"
    r"|acknowledg|agradecimientos|conflictos? de inter[eé]s|financiaci[oó]n"
    r"|author affiliations|corresponding author|submissions?\b|accepted for publication",
    re.IGNORECASE,
)
# Linea con forma de referencia bibliografica ("Smith J, et al. ... 2019;200:
# e45", "doi:10...", "PMID"). En los articulos de revista la bibliografia no
# siempre tiene titulo propio (columnas laterales del Lancet), asi que tambien
# se descarta un fragmento cuando la mayoria de sus lineas son referencias.
_CITATION_LINE_RE = re.compile(
    r"\bet al\b|\bdoi\b|\bpmid\b|\b(19|20)\d{2}\s*;\s*\d+|\b(19|20)\d{2}\s*\.\s*\d+\s*[:(]",
    re.IGNORECASE,
)
_MIN_CITATION_LINES = 4
_CITATION_LINES_RATIO = 0.5


def _looks_like_bibliography(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    hits = sum(1 for line in lines if _CITATION_LINE_RE.search(line))
    if not hits:
        return False
    return hits == len(lines) or (
        hits >= _MIN_CITATION_LINES and hits >= _CITATION_LINES_RATIO * len(lines)
    )


# Encabezados y pies de pagina de revista que se repiten en casi todas las
# paginas ("AMERICAN THORACIC SOCIETY DOCUMENTS", "© 2024 American Medical
# Association. All rights reserved.", "Downloaded from jamanetwork.com by
# ...", la URL de ClinicalKey). Sin quitarlos se colaban en los fragmentos y
# terminaban como titulo de seccion de casi todo el documento. Se detectan
# por documento: una linea que aparece en al menos el 40 % de las paginas
# (y en 3 o mas) es encabezado o pie, no contenido.
_RUNNING_LINE_MIN_PAGES = 3
_RUNNING_LINE_PAGE_RATIO = 0.4
_TABLE_RULE_RE = re.compile(r"^\|?[\s:|-]+\|?$")


def _line_key(line: str) -> str:
    plain = _HTML_TAG_RE.sub("", line)
    plain = re.sub(r"^#+\s*", "", plain.strip())
    return re.sub(r"\s+", " ", re.sub(r"[*_]+", "", plain)).strip().lower()


def running_lines(pages: list[dict]) -> set[str]:
    """Lineas (normalizadas) que se repiten como encabezado o pie de pagina."""
    counts: dict[str, int] = {}
    for page in pages:
        for key in {_line_key(line) for line in (page.get("text") or "").splitlines()}:
            if len(key) >= 6 and not _TABLE_RULE_RE.match(key) and "picture text" not in key:
                counts[key] = counts.get(key, 0) + 1
    threshold = max(_RUNNING_LINE_MIN_PAGES, _RUNNING_LINE_PAGE_RATIO * len(pages))
    return {key for key, n in counts.items() if n >= threshold}


def _drop_running_lines(text: str, keys: set[str]) -> str:
    if not keys:
        return text
    return "\n".join(line for line in text.splitlines() if _line_key(line) not in keys)


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


_NUMBERED_TITLE_RE = re.compile(r"^\d{1,3}\.\s")


def _clean_title(title: str) -> str:
    return _strip_markdown_markers(title).rstrip(" :.").strip()


def _is_upper_title(title: str) -> bool:
    letters = [c for c in title if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


class _TitlePath:
    """Ruta de titulos de una subseccion dentro del protocolo.

    Las subsecciones en negrita ("Hemocultivos", "INDICACIONES DE
    HOSPITALIZACIÓN") pierden el contexto del apartado al que pertenecen
    ("1. RECOMENDACIONES PARA EL DIAGNÓSTICO", "2. RECOMENDACIONES PARA EL
    MANEJO"): sin el, una pregunta por el "manejo inicial" no encontraba
    ningun fragmento de manejo. Niveles, segun el formato de los protocolos:
    - apartado numerado ("2. RECOMENDACIONES PARA EL MANEJO"): reinicia la ruta;
    - titulo en mayusculas ("DIAGNÓSTICO MICROBIOLÓGICO"): cuelga del apartado;
    - titulo en minusculas ("Hemocultivos"): cuelga del ultimo titulo en
      mayusculas, si lo hay, y del apartado.
    """

    def __init__(self) -> None:
        self.top: str | None = None
        self.mid: str | None = None

    def full(self, title: str) -> str:
        clean = _clean_title(title)
        if not clean or clean == "Sin seccion":
            return title
        if _NUMBERED_TITLE_RE.match(clean):
            # "5. Diagnóstico: Análisis de orina": el apartado es "5.
            # Diagnóstico"; "Análisis de orina" es su primera subseccion.
            head, _, rest = clean.partition(":")
            self.top, self.mid = head.strip(), None
            return f"{self.top} — {rest.strip()}" if rest.strip() else self.top
        if _is_upper_title(clean):
            self.mid = clean
            parts = [self.top, clean]
        else:
            parts = [self.top, self.mid, clean]
        return " — ".join(p for p in parts if p)


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


def split_into_sections(
    markdown_text: str, default_title: str = "Sin seccion", keep_empty: bool = False
) -> list[_Section]:
    """Divide un texto markdown en secciones usando encabezados reales (#, ##...),
    etiquetas en negrita al inicio de parrafo ("**Hemocultivos:** ...") y,
    como respaldo (paginas OCR sin markdown), preguntas numeradas en negrita
    al inicio de linea (ej. "**13. En pacientes...**").

    keep_empty conserva los encabezados sin texto (p. ej. un titulo al final
    de la pagina), para saber de que seccion es el texto de la pagina
    siguiente.
    """
    raw_matches: list[tuple[int, int, str]] = [
        (m.start(), m.end(), m.group(2).strip()) for m in _HEADER_RE.finditer(markdown_text)
    ]
    inline = [
        (m.start(), m.end(), _strip_markdown_markers(m.group("label")))
        for m in _INLINE_LABEL_RE.finditer(markdown_text)
        if (m.group("inner") or m.group("outer"))
        and not _CAPTION_RE.match(m.group("label"))
        # "**12.** Smith J, et al." es el numero de una referencia, no un titulo.
        and re.search(r"[^\W\d_]", m.group("label"))
    ]
    inline += [
        (m.start(), m.end(), m.group("label").strip())
        for m in _TABLE_CAPTION_RE.finditer(markdown_text)
    ]
    inline_starts = {start for start, _end, _title in inline}
    raw_matches += inline
    raw_matches += [
        (m.start(), m.end(), _strip_markdown_markers(m.group(1)))
        for m in _BOLD_NUMBERED_HEADER_RE.finditer(markdown_text)
        if m.start() not in inline_starts
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
        if body or keep_empty:
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
    # El texto sin encabezado al inicio de una pagina continua la ultima
    # seccion de la pagina anterior (p. ej. la lista de paneles de Filmarray
    # o los criterios de egreso que siguen en la pagina siguiente). Si esa
    # seccion era la bibliografia, se descarta igual que ella.
    last_title = "Sin seccion"
    path = _TitlePath()
    repeated = running_lines(pages)

    for page in pages:
        text = clean_page_text(_drop_running_lines(page["text"], repeated))
        if page["page"] in drop_tables_on_pages:
            text = strip_markdown_tables(text)
        if not text:
            continue
        sections = split_into_sections(text, default_title=_CONTINUATION, keep_empty=True)
        for section in sections:
            if section.title == _CONTINUATION:
                section.title = last_title
            elif not _is_bibliography(section.title):
                section.title = path.full(section.title)
            last_title = section.title
            if not section.text or _is_bibliography(section.title):
                continue
            if approx_token_count(section.text) <= chunk_size_tokens:
                pieces = [section.text]
            else:
                pieces = _split_long_section(
                    section.text, chunk_size_tokens, chunk_overlap_ratio, min_chunk_tokens
                )
            min_tokens = (
                min(min_chunk_tokens, _MIN_TITLED_SECTION_TOKENS)
                if section.title != "Sin seccion"
                else min_chunk_tokens
            )
            for piece in pieces:
                if approx_token_count(piece) < min_tokens or _looks_like_bibliography(piece):
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
