"""Modo extractivo del RAG: la respuesta se arma con oraciones textuales de
las fuentes, no con texto redactado por el modelo.

Por que: en las pruebas con el modelo local de 7B, la redaccion libre
cambiaba el alcance de los datos aunque el contexto fuera correcto. Con la
pregunta de bacteriuria asintomatica llego a escribir "las guias recomiendan
el tratamiento de la bacteriuria asintomatica en adultos", lo contrario del
protocolo de la CUB (solo gestantes y procedimientos invasivos), con las
condiciones a la vista en el contexto; con las guias externas en ingles
tambien invirtio el sentido de frases y mezclo fuentes.

Como funciona:
1. Cada fragmento recuperado se divide en unidades: oraciones, listas con su
   frase de entrada, o una fila de tabla completa (la condicion de la fila
   nunca se separa de su tratamiento).
2. El modelo solo ELIGE las unidades que responden la pregunta (salida JSON
   restringida a los ids existentes). No escribe texto.
3. Las unidades en ingles se traducen una por una, con instruccion de
   traduccion literal (conservar farmacos, dosis, negaciones y condiciones).
4. La respuesta es la lista de unidades elegidas, cada una con su cita:
   primero las del protocolo institucional en el orden del documento,
   despues las de las fuentes externas. Es una sola respuesta, sin
   encabezados por fuente.

Asi no puede aparecer una frase que no este en alguna fuente: el unico texto
que no es copia es la traduccion, marcada como tal.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from gpc_rag.common.models import RetrievedChunk
from gpc_rag.ingestion.sources import INSTITUTIONAL, source_label

logger = logging.getLogger(__name__)

REFUSAL = (
    "No encuentro informacion suficiente en las guias disponibles para responder "
    "esto con seguridad."
)
TRANSLATED_MARK = "(traducido del inglés)"

_MAX_SENTENCE_WORDS = 70
_MIN_UNIT_WORDS = 4
_BULLET_RE = re.compile(r"^\s*(?:[-•*]|\d{1,2}[.)])\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;])\s+(?=[A-ZÁÉÍÓÚÑ¿(])")
# Titulos de cuadros, tablas o figuras solos ("Box 2. Main Indications to
# Treat Asymptomatic Bacteriuria"): no dicen nada por si mismos.
_CAPTION_UNIT_RE = re.compile(
    r"^(box|table|figure|fig\.|cuadro|tabla|figura|gr[aá]fico)\s*\d+\.?\s+[^.;:]{0,100}\.?$",
    re.IGNORECASE,
)
# El modelo a veces repite el id al traducir ("u3: Caja 2...").
_LEADING_ID_RE = re.compile(r"^\s*u\d+\s*[:.-]\s*", re.IGNORECASE)
_KEY_VALUE_RE = re.compile(r"^[^:\n]{2,60}:\s*\S")
_MARKDOWN_RE = re.compile(r"(\*\*|__|(?<!\w)[*_](?!\s)|(?<!\s)[*_](?!\w)|^#+\s*)", re.MULTILINE)

_EN_WORDS = frozenset({
    "the", "of", "and", "to", "in", "is", "for", "with", "on", "as", "by", "be",
    "are", "was", "were", "this", "that", "these", "those", "or", "an", "at",
    "from", "it", "its", "which", "should", "patients", "treatment", "recommend",
    "recommended",
})
_ES_WORDS = frozenset({
    "el", "la", "de", "que", "los", "las", "para", "con", "por", "del", "se", "en",
    "una", "un", "es", "son", "al", "como", "pacientes", "tratamiento", "debe",
    "recomienda",
})


@dataclass
class Unit:
    uid: str
    text: str
    source: RetrievedChunk
    rank: int  # posicion del fragmento en la recuperacion
    order: int  # posicion de la unidad dentro del fragmento

    @property
    def english(self) -> bool:
        return is_english(self.text)


def is_english(text: str) -> bool:
    words = re.findall(r"[a-záéíóúñ]+", text.lower())
    en = sum(1 for w in words if w in _EN_WORDS)
    es = sum(1 for w in words if w in _ES_WORDS)
    return en > es


def _clean(text: str) -> str:
    text = _MARKDOWN_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _last_level(section: str) -> str:
    return _clean(section.split(" — ")[-1]).rstrip(" :.")


def _body(rc: RetrievedChunk) -> str:
    """Texto del fragmento sin el titulo de seccion que se le antepone al
    indexar (ingestion/chunking.py) ni el encabezado de fila de tabla."""
    text = rc.chunk.text.strip()
    first, _, rest = text.partition("\n")
    head = _clean(first).rstrip(" :.").lower()
    section = _clean(rc.chunk.section).rstrip(" :.").lower()
    if rest and head and (head == section or section.startswith(head) or head.startswith(section)):
        return rest.strip()
    return text


def _is_table_row(body: str) -> bool:
    lines = [line for line in body.splitlines() if line.strip()]
    if not lines:
        return False
    key_values = sum(1 for line in lines if _KEY_VALUE_RE.match(_clean(line)))
    return key_values >= 3 and key_values >= 0.5 * len(lines)


def _split_long(paragraph: str) -> list[str]:
    if len(paragraph.split()) <= _MAX_SENTENCE_WORDS:
        return [paragraph]
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]


def segment(rc: RetrievedChunk) -> list[str]:
    """Unidades de texto de un fragmento, en su orden."""
    body = _body(rc)
    if _is_table_row(body):
        # Fila de tabla: una sola unidad, para que la condicion de la fila no
        # se separe de su tratamiento ni de sus notas.
        row = [_clean(line) for line in body.splitlines() if line.strip()]
        heading = _clean(rc.chunk.section)
        return [f"{heading}. " + " ".join(r if r.endswith(".") else r + "." for r in row)]

    units: list[str] = []
    bullets: list[str] = []
    lead: str | None = None

    def flush() -> None:
        nonlocal bullets, lead
        if bullets:
            prefix = lead.rstrip(" :") if lead else _last_level(rc.chunk.section)
            units.append(f"{prefix}: " + "; ".join(b.rstrip(" .;") for b in bullets) + ".")
        bullets, lead = [], None

    for paragraph in re.split(r"\n\s*\n", body):
        for line in paragraph.splitlines() if _BULLET_RE.match(paragraph) else [paragraph]:
            if _BULLET_RE.match(line):
                bullets.append(_clean(_BULLET_RE.sub("", line)))
                continue
            flush()
            cleaned = _clean(line)
            if not cleaned:
                continue
            if cleaned.endswith(":") and len(cleaned.split()) <= 30:
                lead = cleaned  # frase de entrada de la lista que sigue
                continue
            units.extend(_split_long(cleaned))
    if lead and not bullets:
        units.append(lead)
    flush()
    return [
        u for u in units if len(u.split()) >= _MIN_UNIT_WORDS and not _CAPTION_UNIT_RE.match(u)
    ]


def build_units(sources: list[RetrievedChunk]) -> list[Unit]:
    units: list[Unit] = []
    seen: set[str] = set()
    for rank, rc in enumerate(sources):
        for order, text in enumerate(segment(rc)):
            key = text.lower()
            if key in seen:  # solape entre fragmentos consecutivos
                continue
            seen.add(key)
            units.append(Unit(uid=f"u{len(units) + 1}", text=text, source=rc, rank=rank, order=order))
    return units


_SELECT_SYSTEM = """\
Eres un selector de evidencia clinica. Recibes una PREGUNTA y una lista de \
ORACIONES numeradas, copiadas de protocolos y guias clinicas. Tu unico \
trabajo es elegir los ids de las oraciones que responden la pregunta. No \
respondas, no resumas y no reescribas nada.

Reglas:
- Elige solo oraciones que respondan directamente lo que se pregunta.
- Si eliges una recomendacion o un tratamiento, elige tambien la oracion que \
dice a que pacientes o situacion aplica (condicion, poblacion, excepciones), \
si esta en la lista.
- Si la pregunta es sobre cuando hacer algo (tratar, hospitalizar, pedir un \
examen), elige las oraciones que dan esas indicaciones antes que las que \
solo dan dosis.
- Prefiere las oraciones del protocolo institucional de la Clinica \
Universitaria Bolivariana (CUB) cuando traten el tema; agrega las de guias \
externas o revisiones que lo complementen.
- No elijas oraciones repetidas ni de otro tema.
- Si ninguna oracion responde la pregunta, devuelve la lista vacia.

Responde UNICAMENTE con JSON: {"ids": ["u3", "u7", ...]}"""

_TRANSLATE_SYSTEM = """\
Traduce al español cada texto, de forma literal y completa. Conserva \
exactamente los nombres de farmacos, dosis, unidades, numeros, intervalos, \
duraciones, negaciones ("no", "not") y condiciones. Usa la terminologia \
medica estandar en español (p. ej. pyelonephritis = pielonefritis, \
asymptomatic bacteriuria = bacteriuria asintomatica, cystitis = cistitis, \
UTI = ITU). No agregues, quites ni expliques nada, y no incluyas el id en \
el texto traducido. Responde UNICAMENTE con JSON: {"traducciones": {"<id>": \
"<texto traducido>", ...}}"""


def _source_header(rc: RetrievedChunk) -> str:
    chunk = rc.chunk
    name = chunk.source_title or chunk.source_file
    return f"[{source_label(chunk.source_type)} | {name} | {chunk.section} | p. {chunk.page}]"


def _selection_prompt(question: str, units: list[Unit], extra: str | None) -> str:
    lines = [f"PREGUNTA:\n{question}\n", "ORACIONES:"]
    current = None
    for unit in units:
        if unit.source is not current:
            current = unit.source
            lines.append(_source_header(current))
        lines.append(f"  {unit.uid}: {unit.text}")
    if extra:
        lines.append(f"\nINSTRUCCION ADICIONAL:\n{extra}")
    return "\n".join(lines)


def _selection_schema(units: list[Unit], max_units: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "ids": {
                "type": "array",
                "items": {"type": "string", "enum": [u.uid for u in units]},
                "maxItems": max_units,
            }
        },
        "required": ["ids"],
    }


def select_units(
    generator, question: str, units: list[Unit], max_units: int, extra: str | None = None
) -> list[Unit]:
    if not units:
        return []
    content = generator.chat_json(
        _SELECT_SYSTEM,
        _selection_prompt(question, units, extra),
        _selection_schema(units, max_units),
        num_predict=20 + 8 * max_units,
    )
    try:
        ids = json.loads(content).get("ids") or []
    except (json.JSONDecodeError, AttributeError):
        logger.warning("RAG extractivo: seleccion no interpretable: %r", content)
        return []
    by_id = {u.uid: u for u in units}
    chosen: list[Unit] = []
    for uid in ids:
        unit = by_id.get(uid)
        if unit is not None and unit not in chosen:
            chosen.append(unit)
    return chosen[:max_units]


def translate_units(generator, units: list[Unit]) -> dict[str, str]:
    """Traduccion de las unidades en ingles ({uid: texto}); las que fallen se
    dejan en su idioma original."""
    english = [u for u in units if u.english]
    if not english:
        return {}
    schema = {
        "type": "object",
        "properties": {
            "traducciones": {
                "type": "object",
                "properties": {u.uid: {"type": "string"} for u in english},
                "required": [u.uid for u in english],
            }
        },
        "required": ["traducciones"],
    }
    prompt = "\n".join(f"{u.uid}: {u.text}" for u in english)
    content = generator.chat_json(
        _TRANSLATE_SYSTEM,
        prompt,
        schema,
        num_predict=sum(2 * len(u.text.split()) + 20 for u in english),
    )
    try:
        translations = json.loads(content).get("traducciones") or {}
    except (json.JSONDecodeError, AttributeError):
        logger.warning("RAG extractivo: traduccion no interpretable: %r", content)
        return {}
    return {
        uid: _LEADING_ID_RE.sub("", text).strip()
        for uid, text in translations.items()
        if isinstance(text, str) and text.strip() and uid in {u.uid for u in english}
    }


def _citation(rc: RetrievedChunk) -> str:
    c = rc.chunk
    return f"[Guia: {c.source_file}, Seccion: {c.section}, Pagina: {c.page}]"


def _display_order(unit: Unit) -> tuple:
    chunk = unit.source.chunk
    if chunk.source_type == INSTITUTIONAL:
        # El protocolo en el orden del documento: las indicaciones (pag. 3)
        # antes que las tablas de tratamiento (pag. 4).
        return (0, chunk.source_file, chunk.page or 0, chunk.chunk_index, unit.order)
    return (1, unit.rank, unit.order, "", 0)


def compose(units: list[Unit], translations: dict[str, str]) -> str:
    if not units:
        return REFUSAL
    lines = []
    for unit in sorted(units, key=_display_order):
        text = translations.get(unit.uid)
        shown = f"{text} {TRANSLATED_MARK}" if text else unit.text
        lines.append(f"- {shown} {_citation(unit.source)}")
    return "\n".join(lines)


def extractive_answer(
    generator,
    question: str,
    sources: list[RetrievedChunk],
    max_units: int = 8,
    extra_instruction: str | None = None,
) -> str:
    units = build_units(sources)
    chosen = select_units(generator, question, units, max_units, extra_instruction)
    logger.info(
        "RAG extractivo: %d unidades candidatas, elegidas=%s",
        len(units),
        [u.uid for u in chosen],
    )
    return compose(chosen, translate_units(generator, chosen))
