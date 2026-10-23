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
_MAX_LIST_WORDS = 80
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


# Llamadas numericas de los articulos ("[6]", "[12, 13]") y vinetas sueltas
# dentro de una linea ("embarazo; • Procedimientos").
_NUMERIC_CALLOUT_RE = re.compile(r"\s*\[\d+(?:\s*[,–-]\s*\d+)*\]")
_INLINE_BULLET_RE = re.compile(r"\s*[•➢▪◦]\s*")


# Formato de autoevaluacion de AJKD ("..., so the answer is (a).").
_QUIZ_RE = re.compile(
    r",?\s*(so|therefore|thus)?\s*the (correct )?answer is \(?[a-e]\)?\.?|\bpor lo que la respuesta es \(?[a-e]\)?\.?",
    re.IGNORECASE,
)


# Numeros de referencia que ClinicalKey deja como texto al final de cada
# frase ("...nonpregnant adults 1 52 53;"). Solo en fuentes externas y solo
# cuando son dos o mas seguidos tras una palabra: un numero solo al final
# ("older than 65.") puede ser un dato.
_TRAILING_REFS_RE = re.compile(r"(?<=[a-z)])(?:\s\d{1,3}){2,4}(?=\s*(?:[;.]|$))")


def _clean(text: str) -> str:
    text = _QUIZ_RE.sub(".", text)
    text = re.sub(r"\.\s*\.", ".", text)
    text = _MARKDOWN_RE.sub("", text)
    text = _NUMERIC_CALLOUT_RE.sub("", text)
    text = _INLINE_BULLET_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _table_row_unit(body: str, section: str) -> str:
    """Una fila de tabla en una linea legible: "Encabezado. Col: valor. Col:
    valor. Notas: (1) ...; (3) ..." (sin repetir el encabezado). Las celdas
    se separan con punto porque sus valores ya usan ";" entre farmacos."""
    cells: list[str] = []
    notes: list[str] = []
    in_notes = False
    for raw in body.splitlines():
        line = _clean(raw)
        if not line:
            continue
        if line.lower().startswith("notas de la tabla"):
            in_notes = True
            continue
        if in_notes:
            notes.append(_BULLET_RE.sub("", line).rstrip(" ."))
        elif line.rstrip(" .") != _clean(section).rstrip(" ."):
            cells.append(line.rstrip(" ."))
    text = f"{_clean(section)}. " + ". ".join(cells) + "."
    if notes:
        text += " Notas: " + "; ".join(notes) + "."
    return text


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
        return [_table_row_unit(body, rc.chunk.section)]

    units: list[str] = []
    bullets: list[str] = []
    lead: str | None = None

    def flush() -> None:
        nonlocal bullets, lead
        if bullets:
            prefix = lead.rstrip(" :") if lead else _last_level(rc.chunk.section)
            items = [b.rstrip(" .;") for b in bullets]
            joined = f"{prefix}: " + "; ".join(items) + "."
            if len(joined.split()) <= _MAX_LIST_WORDS:
                units.append(joined)
            else:
                # Lista larga: un elemento por unidad, cada uno con su frase de
                # entrada (una sola unidad de 300 palabras no se traducia).
                units.extend(f"{prefix}: {item}." for item in items)
        bullets, lead = [], None

    for paragraph in re.split(r"\n\s*\n", body):
        lines = paragraph.splitlines()
        has_bullets = any(_BULLET_RE.match(line) for line in lines)
        for line in lines if has_bullets else [paragraph]:
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
        u
        for u in _attach_anaphora(units)
        if len(u.split()) >= _MIN_UNIT_WORDS and not _CAPTION_UNIT_RE.match(u)
    ]


# Oracion que remite a la anterior ("No contamos con datos locales para
# validar este hallazgo...", "These patients..."): sola pierde de que habla.
_ANAPHORA_RE = re.compile(
    r"^(este|esta|estos|estas|esto|dicho|dicha|dichos|dichas|ello|en ellos|en ellas"
    r"|this|these|those|such|it|they)\b|\b(este|esta|estos|estas) (hallazgo|resultado|"
    r"grupo|caso|escenario|enfoque|recomendaci[oó]n)\b",
    re.IGNORECASE,
)


def _attach_anaphora(units: list[str]) -> list[str]:
    """Une a la oracion anterior las que remiten a ella."""
    merged: list[str] = []
    for unit in units:
        if merged and _ANAPHORA_RE.search(unit[:80]):
            merged[-1] = f"{merged[-1]} {unit}"
        else:
            merged.append(unit)
    return merged


def build_units(sources: list[RetrievedChunk]) -> list[Unit]:
    units: list[Unit] = []
    seen: set[str] = set()
    for rank, rc in enumerate(sources):
        for order, raw in enumerate(segment(rc)):
            text = (
                _TRAILING_REFS_RE.sub("", raw) if rc.chunk.source_type != INSTITUTIONAL else raw
            )
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
- Elige las oraciones que respondan directamente lo que se pregunta \
(normalmente entre 2 y 6), y ninguna que no lo haga.
- No elijas oraciones que describan el documento, su panel, su proposito, su \
alcance o su metodo ("el panel abordo...", "esta guia tiene como \
proposito..."): solo las que dicen que hacer con el paciente.
- Si la pregunta es sobre una enfermedad o condicion que ninguna oracion \
trata (p. ej. malaria cuando las oraciones hablan de neumonia o de ITU), \
devuelve la lista vacia aunque haya oraciones sobre antibioticos o fiebre.
- No elijas oraciones especulativas o sobre necesidades de investigacion \
("podrian", "no contamos con datos") si hay recomendaciones concretas.
- Si eliges una recomendacion o un tratamiento, elige tambien la oracion que \
dice a que pacientes o situacion aplica (condicion, poblacion, excepciones), \
si esta en la lista.
- Si la pregunta es sobre cuando hacer algo (tratar, hospitalizar, pedir un \
examen), elige las oraciones que dan esas indicaciones antes que las que \
solo dan dosis.
- Prefiere las oraciones del protocolo institucional de la Clinica \
Universitaria Bolivariana (CUB) cuando traten el tema; agrega las de guias \
externas o revisiones que lo complementen.
- No elijas oraciones repetidas ni de otro tema o situacion clinica distinta \
a la preguntada (p. ej. choque septico si la pregunta es sobre bacteriuria \
asintomatica), aunque vengan del mismo documento.
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


_TRANSLATE_BATCH = 3


def translate_units(generator, units: list[Unit]) -> dict[str, str]:
    """Traduccion de las unidades en ingles ({uid: texto}), de a pocas por
    llamada; las que fallen se dejan en su idioma original."""
    english = [u for u in units if u.english]
    translations: dict[str, str] = {}
    for start in range(0, len(english), _TRANSLATE_BATCH):
        translations.update(_translate_batch(generator, english[start : start + _TRANSLATE_BATCH]))
    return translations


def _translate_batch(generator, english: list[Unit]) -> dict[str, str]:
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
        num_predict=sum(3 * len(u.text.split()) + 40 for u in english),
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
        shown = f"{_clean(text)} {TRANSLATED_MARK}" if text else unit.text
        lines.append(f"- {shown} {_citation(unit.source)}")
    return "\n".join(lines)


def extractive_answer(
    generator,
    question: str,
    sources: list[RetrievedChunk],
    max_units: int = 8,
    extra_instruction: str | None = None,
    min_relevance: float = 0.0,
) -> str:
    if not sources or max(rc.score for rc in sources) < min_relevance:
        # Ningun fragmento le parecio pertinente al reranker (p. ej. malaria):
        # abstencion sin pedirle al selector que elija entre oraciones ajenas.
        logger.info("RAG extractivo: ningun fragmento supera %.2f; abstencion.", min_relevance)
        return REFUSAL
    units = build_units(sources)
    chosen = select_units(generator, question, units, max_units, extra_instruction)
    logger.info(
        "RAG extractivo: %d unidades candidatas, elegidas=%s",
        len(units),
        [u.uid for u in chosen],
    )
    return compose(chosen, translate_units(generator, chosen))
