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
from dataclasses import dataclass, replace

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
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;])\s+(?=[A-ZÁÉÍÓÚÑ¿(])|\s*(?=[†‡§¶])")
# Notas al pie de figuras ("...MRSA.8 †Recomendaciones... ‡Señales...") y
# referencias pegadas al punto ("MRSA.8") en las fuentes externas.
_FOOTNOTE_MARK_RE = re.compile(r"[†‡§¶]")
_DOT_REF_RE = re.compile(r"(?<=[A-Za-z)])\.(\d{1,3}(?:[,–-]\d{1,3})*)(?=\s|$)")
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


def _is_table_row(rc: RetrievedChunk, body: str) -> bool:
    """Fila del archivo de tablas revisadas (ingestion/tables.py): su texto
    empieza con su encabezado y un punto. Sin esa condicion, la lista de
    Definiciones del protocolo ("Termino: definicion") se tomaba por fila y
    entraba completa como una sola oracion."""
    if not rc.chunk.text.startswith(f"{rc.chunk.section}."):
        return False
    lines = [line for line in body.splitlines() if line.strip()]
    if not lines:
        return False
    key_values = sum(1 for line in lines if _KEY_VALUE_RE.match(_clean(line)))
    return key_values >= 3 and key_values >= 0.5 * len(lines)


def _split_long(paragraph: str) -> list[str]:
    if len(paragraph.split()) <= _MAX_SENTENCE_WORDS and not _FOOTNOTE_MARK_RE.search(paragraph):
        return [paragraph]
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]


def segment(rc: RetrievedChunk) -> list[str]:
    """Unidades de texto de un fragmento, en su orden."""
    body = _body(rc)
    if _is_table_row(rc, body):
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
                item = _clean(_BULLET_RE.sub("", line))
                if bullets and item[:1].islower():
                    # Renglon partido por el extractor ("...estructurales o" /
                    # "- funcionales de las vias urinarias"): continua el anterior.
                    bullets[-1] = f"{bullets[-1].rstrip(' .')} {item}"
                else:
                    bullets.append(item)
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


# Referencias bibliograficas dentro de una seccion clinica (las "Additional
# readings" del Core Curriculum de AJKD van al pie de cada apartado): "Nicolle
# LE, Gupta K, et al. Clinical practice guideline... doi:...". No son
# recomendaciones; el selector llego a elegir una para bacteriuria asintomatica.
_CITATION_UNIT_RE = re.compile(
    r"\bet al\b|doi\.org|\bdoi\s*:|https?://|\bpmid\b|essential reading", re.IGNORECASE
)
# Inicio de una referencia: "Nicolle LE, Gupta K" / "Smith J et al".
_REFERENCE_START_RE = re.compile(r"(?:^|(?<=[.;:]\s))[A-Z][\w'-]+ [A-Z]{1,3}(?:,| et al)")


def _strip_references(text: str) -> str:
    """Quita la cola bibliografica de una unidad; vacio si toda es referencia."""
    if not _CITATION_UNIT_RE.search(text):
        return text
    m = _REFERENCE_START_RE.search(text)
    head = text[: m.start()].strip() if m else ""
    # Solo queda la etiqueta de la seccion ("Asymptomatic Bacteriuria:").
    return head if len(head.split()) >= 4 and not head.endswith(":") else ""


# Resumen de un estudio en la discusion de la evidencia de una guia ("El
# estudio de Prina introdujo...", "El analisis de Metersky y colaboradores..."):
# es el respaldo de una recomendacion, no la recomendacion. En "factores para
# decidir el manejo de la neumonia" el selector las elegia en lugar de
# recomendaciones. Un OR o un IC95 solos no bastan: la GPC tambien los usa al
# enunciar factores de riesgo ("aislamiento previo de P. aeruginosa (OR=14.2)").
_STUDY_SUMMARY_RE = re.compile(
    r"\b(?:[Ee]l|[Uu]n|[Ll]os|[Ee]ste|[Dd]icho)\s+(?:estudio|an[aá]lisis|ensayo|metan[aá]lisis)s?\s+de\s+[A-Z]"
    r"|\by colaboradores\b"
)
# Llamadas de referencia pegadas en cursiva markdown ("_SAMR_157", "_P.
# aeruginosa_ 132") y espacios antes de puntuacion ("Pseudomonas ,").
_UNDERSCORE_REF_RE = re.compile(r"_\s?\d{1,3}(?:[,–-]\d{1,3})*\b")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:])")


def _tidy_external(text: str) -> str:
    text = _UNDERSCORE_REF_RE.sub("", text).replace("_", "")
    return _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text).strip()


# Fragmentos cortados por un salto de columna o de pagina ("No single clinical
# sign has good predictive") y leyendas que son listas de abreviaturas
# ("ATS=..., FiO2=..., IDSA=..."): no se pueden citar como recomendacion.
_ABBREVIATION_DEF_RE = re.compile(r"\b(?!(?:OR|HR|RR)\b)[A-Z][A-Za-z0-9]{1,8}\s?=\s?[A-Za-z]")
_TERMINAL_PUNCT = tuple(".!?:;)\"»")


# Verbo conjugado: una oracion (no un item de lista como "Frailty or poor
# social support") que no termina en puntuacion quedo cortada.
_FINITE_VERB_RE = re.compile(
    r"\b(is|are|was|were|has|have|had|should|can|may|must|recommends?|suggests?"
    r"|es|son|fue|tiene|tienen|debe|deben|puede|pueden|recomienda|recomendamos|sugerimos)\b",
    re.IGNORECASE,
)


# Palabra de enlace al final: la oracion sigue en otro lado ("...within the",
# "...we suggest", "...con menor").
_DANGLING_WORDS = frozenset(
    """the a an of to in for with and or but than that which who we be by from at as
    plus less more greater el la los las un una de del al en con por para y o que
    se su sus menor mayor""".split()  # noqa: SIM905 -- lista legible
)


# Atribucion de una figura o tabla ("Advice from the 2025 American Thoracic
# Society guideline on...", "Tomada y adaptada de...") y definicion suelta de
# una abreviatura ("MRSA=meticillin-resistant Staphylococcus aureus.").
_ATTRIBUTION_RE = re.compile(
    r"^(advice|adapted|reproduced|modified|data|source|tomad[oa]|adaptad[oa])\b.{0,40}\b(from|de)\b",
    re.IGNORECASE,
)
_LEADING_ABBREVIATION_RE = re.compile(r"^[A-Z][\w ]{0,15}?\s?=\s?\S")


_LAST_SENTENCE_END_RE = re.compile(r"[.!?](?=\s+[A-ZÁÉÍÓÚÑ¿¡])")


def _trim_unfinished_tail(text: str) -> str:
    """Quita la ultima oracion de una unidad larga si quedo cortada ("...que
    tienen factores de riesgo como"), cuando antes hay oraciones completas."""
    body = text.rstrip(" |")
    if body.endswith(_TERMINAL_PUNCT):
        return text
    ends = list(_LAST_SENTENCE_END_RE.finditer(body))
    if not ends:
        return text
    tail = body[ends[-1].end() :].split()
    if tail and (tail[-1].lower() in _DANGLING_WORDS or len(tail) < 15):
        return body[: ends[-1].end()].strip()
    return text


def _unusable_external(text: str) -> bool:
    body = text.rstrip(" |")
    body = re.sub(r"[\d,.\s]+$", lambda m: "." if "." in m.group(0) else "", body)
    if len(_ABBREVIATION_DEF_RE.findall(text)) >= 3:
        return True
    if _ATTRIBUTION_RE.match(text) or _LEADING_ABBREVIATION_RE.match(text):
        return True
    if text[:1].islower() and len(body.split()) < 15:
        return True  # continuacion de una oracion de otro fragmento
    if body.endswith(_TERMINAL_PUNCT):
        return False
    words = body.split()
    # Solo se descartan los cortes cortos y evidentes. Una unidad larga se
    # conserva aunque no tenga punto final: puede ser una recomendacion
    # completa ("Punto de buena practica ... se recomienda adicionar terapia
    # empirica con macrolido") o una cortada al final de un fragmento que aun
    # asi trae la recomendacion.
    if len(words) >= 15:
        return False
    return (
        text[:1].islower()
        or words[-1].lower() in _DANGLING_WORDS
        or (len(words) < 8 and bool(_FINITE_VERB_RE.search(body)))
    )


def build_units(sources: list[RetrievedChunk]) -> list[Unit]:
    units: list[Unit] = []
    seen: set[str] = set()
    for rank, rc in enumerate(sources):
        for order, raw in enumerate(segment(rc)):
            text = raw
            if rc.chunk.source_type != INSTITUTIONAL:
                text = _DOT_REF_RE.sub(".", _TRAILING_REFS_RE.sub("", text))
                text = _tidy_external(_FOOTNOTE_MARK_RE.sub("", text))
            text = _strip_references(text)
            if not text:
                continue
            if rc.chunk.source_type != INSTITUTIONAL:
                text = _trim_unfinished_tail(text)
                if _STUDY_SUMMARY_RE.search(text) or _unusable_external(text):
                    continue
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
- Las preguntas sobre el enfoque, el manejo o cuando hacer algo se responden \
con acciones concretas (criterios, indicaciones, examenes, tratamientos), no \
con frases generales ("su abordaje se centra en...", "las ITU son un motivo \
frecuente de consulta..."). Si el protocolo institucional trae esas acciones, \
eligelas.
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
- Si la pregunta es sobre el manejo inicial, elige primero lo que decide \
donde tratar al paciente (indicaciones de hospitalizacion o de UCI, escalas de \
gravedad) y el antibiotico inicial; despues, si cabe, los examenes. No elijas \
lo del seguimiento, las complicaciones o el largo plazo.
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
UTI = ITU, screening = tamizaje, culture = cultivo, culture susceptibility = \
antibiograma). No agregues, quites ni expliques nada, y no incluyas el id en \
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


def merge_continuations(sources: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Une los fragmentos consecutivos de una misma seccion (chunk_index n y
    n+1). Una lista partida entre paginas ("Indicaciones para paso a terapia
    oral y/o egreso": 48 horas afebril y TAS > 90 en la pagina 3, el resto en
    la 4) quedaba en dos fragmentos, y el selector elegia solo uno: a la
    respuesta le faltaban criterios."""
    by_key = {(rc.chunk.source_file, rc.chunk.section, rc.chunk.chunk_index): rc for rc in sources}
    merged: list[RetrievedChunk] = []
    for rc in sources:
        key = (rc.chunk.source_file, rc.chunk.section)
        if (*key, rc.chunk.chunk_index - 1) in by_key:
            continue  # se une al fragmento anterior de la misma seccion
        text, score, index = rc.chunk.text, rc.score, rc.chunk.chunk_index
        while (nxt := by_key.get((*key, index + 1))) is not None:
            text = f"{text}\n\n{_body(nxt)}"
            score = max(score, nxt.score)
            index += 1
        if index == rc.chunk.chunk_index:
            merged.append(rc)
        else:
            merged.append(replace(rc, chunk=replace(rc.chunk, text=text), score=score))
    return merged


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
    units = build_units(merge_continuations(sources))
    institutional = [u for u in units if u.source.chunk.source_type == INSTITUTIONAL]
    external = [u for u in units if u.source.chunk.source_type != INSTITUTIONAL]
    # Seleccion por separado: primero el protocolo institucional, despues las
    # fuentes externas. En una sola seleccion, ante "¿enfoque recomendado
    # para el manejo inicial de la NAC?" el modelo eligio dos frases
    # generales del resumen de la GPC 2026 y ninguna del protocolo, aunque
    # sus indicaciones de hospitalizacion estaban en la lista.
    inst_max = max_units // 2 + max_units % 2 if external else max_units
    chosen = select_units(generator, question, institutional, inst_max, extra_instruction)
    chosen += select_units(
        generator, question, external, max_units - len(chosen), extra_instruction
    ) if external and len(chosen) < max_units else []
    logger.info(
        "RAG extractivo: %d unidades candidatas (%d del protocolo), elegidas=%s",
        len(units),
        len(institutional),
        [u.uid for u in chosen],
    )
    return compose(chosen, translate_units(generator, chosen))
