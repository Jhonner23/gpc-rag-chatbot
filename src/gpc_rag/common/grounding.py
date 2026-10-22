"""Controles deterministas de fundamentacion (sin modelos de lenguaje).

Complementan a los jueces locales (evaluador del RAG y verificacion de
fidelidad del integrador). En pruebas reales con qwen2.5 en CPU, los jueces
aprobaron respuestas que citaban documentos inexistentes ("Guia ... Poblacion
Gestante") y que comparaban contra "otras guias" cuando todo el contexto venia
de un solo protocolo. Estas dos fallas se pueden detectar sin un modelo:

- Citas desconocidas: toda cita con el formato [Guia: <archivo>, ...] debe
  nombrar un archivo que realmente se recupero (o el arbol clinico).
- Guias fantasma: si todas las fuentes vienen de un mismo documento, la
  respuesta no puede hablar de "otras guias" o "algunas guias".
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

_CITATION_RE = re.compile(r"\[\s*gu[ií]a\s*:\s*([^,\]]+)", re.IGNORECASE)
_OTHER_GUIDES_RE = re.compile(
    r"\b(otras|algunas|diferentes|distintas|demas|varias)\s+(gu[ií]as|fuentes|protocolos)\b",
    re.IGNORECASE,
)
# "Todas las fuentes coinciden", "ambas guias", "las guias coinciden":
# comparaciones entre varios documentos que no existen cuando solo hay uno.
_SEVERAL_GUIDES_RE = re.compile(
    r"\b(ambas|las dos)\s+(gu[ií]as|fuentes|protocolos)\b"
    r"|\b(gu[ií]as|fuentes|protocolos)\s+(\w+\s+){0,2}(coinciden|concuerdan)\b"
    r"|\b(gu[ií]as|fuentes|protocolos)\b(\s+\w+){0,8}\s+(presentan|dan|tienen|ofrecen|hacen)\s+las\s+mismas\b",
    re.IGNORECASE,
)
_CODE_RE = re.compile(r"CT-PL-\d+", re.IGNORECASE)


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", (text or "").lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _matches(cited: str, allowed: str) -> bool:
    cited_n, allowed_n = _norm(cited), _norm(allowed)
    if not cited_n:
        return False
    stem = _norm(re.sub(r"\.[a-z0-9]+$", "", allowed, flags=re.IGNORECASE))
    return (
        cited_n in {allowed_n, stem}
        or allowed_n.startswith(cited_n)
        or stem.startswith(cited_n)
    )


def unknown_citations(text: str, allowed_sources: Iterable[str]) -> list[str]:
    """Nombres citados con [Guia: ...] que no corresponden a ninguna fuente."""
    allowed = [s for s in allowed_sources if s]
    return [
        cited.strip()
        for cited in _CITATION_RE.findall(text or "")
        if not any(_matches(cited, source) for source in allowed)
    ]


_NEGATION_RE = re.compile(
    r"(no hay|no existen|ninguna|sin|no se encontraron)\s+(\w+\s+){0,2}$"
)


def _document_key(source: str) -> str:
    code = _CODE_RE.search(source)
    return code.group(0).upper() if code else source.strip().lower()


def mentions_other_guides(text: str, sources: Iterable[str]) -> bool:
    """True si habla de otras guias cuando todas las fuentes son un mismo
    documento. Se ignoran las negaciones ("no hay otras guias")."""
    documents = {_document_key(s) for s in sources if s}
    if len(documents) > 1:
        return False
    plain = _norm_keep_spaces(text)
    if _SEVERAL_GUIDES_RE.search(plain):
        return True
    for match in _OTHER_GUIDES_RE.finditer(plain):
        before = plain[max(0, match.start() - 30) : match.start()]
        if not _NEGATION_RE.search(before):
            return True
    return False


def _norm_keep_spaces(text: str) -> str:
    text = unicodedata.normalize("NFKD", (text or "").lower())
    return "".join(c for c in text if not unicodedata.combining(c))


# Deducciones del modelo presentadas como respuesta ("se puede inferir que se
# recomienda...", "el enfoque probablemente incluiria..."). El prompt las
# prohibe, pero el evaluador local las aprobo. Ninguna de estas expresiones
# aparece en los protocolos indexados (comprobado sobre el texto de los PDF,
# las tablas y los arboles).
_INFERENCE_RE = re.compile(
    r"\b(se puede inferir|puede inferirse|se infiere|se puede deducir|puede deducirse"
    r"|se deduce|probablemente|posiblemente)\b",
    re.IGNORECASE,
)


def inference_markers(text: str) -> list[str]:
    plain = _norm_keep_spaces(text)
    return sorted({m.group(0) for m in _INFERENCE_RE.finditer(plain)})


def grounding_problems(text: str, sources: Iterable[str]) -> list[str]:
    """Problemas de fundamentacion detectables sin modelo (vacio si ninguno)."""
    sources = list(sources)
    problems = []
    unknown = unknown_citations(text, sources)
    if unknown:
        problems.append(
            "Cita documentos que no estan entre las fuentes recuperadas: "
            + "; ".join(unknown)
            + "."
        )
    markers = inference_markers(text)
    if markers:
        problems.append(
            "Presenta deducciones propias como respuesta ("
            + ", ".join(f'"{m}"' for m in markers)
            + "): solo puede decir lo que el contexto establece."
        )
    if mentions_other_guides(text, sources):
        problems.append(
            "Menciona otras guias o fuentes, pero todo el contexto proviene de un solo documento."
        )
    return problems
