"""Catalogo de las fuentes externas del RAG (data/01_raw/externas/fuentes.yaml).

Los protocolos institucionales de la CUB (data/01_raw/gpc/) son la
referencia; las guias externas y las revisiones que envio la clinica los
complementan. Cada PDF externo debe estar en el catalogo con su tipo
("guia" o "revision"), tema y titulo: el tipo viaja con cada fragmento para
que la respuesta diga de que clase de fuente sale cada dato, y para reservar
lugares al protocolo institucional en la recuperacion (retrieval/hybrid.py).

Los PDF externos son de editoriales (IDSA, ATS, JAMA, NEJM, Lancet,
ClinicalKey...) y NO se versionan: el repositorio es publico. Solo se
versiona este catalogo.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import yaml

INSTITUTIONAL = "institucional"
EXTERNAL_TYPES = ("guia", "revision")
CATALOG_FILENAME = "fuentes.yaml"

# Como se nombra cada tipo de fuente ante el modelo y en el chat.
SOURCE_TYPE_LABELS = {
    # Con "CUB" a secas el modelo llego a llamarlo "protocolo cubano".
    INSTITUTIONAL: "protocolo institucional de la Clínica Universitaria Bolivariana (CUB)",
    "guia": "guía externa",
    "revision": "revisión",
}


@dataclass(frozen=True)
class SourceInfo:
    file: str
    source_type: str
    topic: str
    title: str
    year: int | None = None


def _file_key(name: str) -> str:
    """Nombre de archivo comparable: sin tildes, mayusculas ni signos (los
    ZIP de Windows/Drive cambian la codificacion de las tildes: unzip deja
    "elecci#U00f3n" en vez de "elección")."""
    name = re.sub(r"#U([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), name)
    text = unicodedata.normalize("NFKD", name.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", text)


def load_catalog(directory: Path) -> dict[str, SourceInfo]:
    """Catalogo de la carpeta, indexado por nombre de archivo normalizado."""
    path = Path(directory) / CATALOG_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Falta {path}: cada PDF externo debe declarar su tipo, tema y titulo."
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    catalog: dict[str, SourceInfo] = {}
    for raw in data.get("documentos") or []:
        source_type = str(raw.get("tipo", "")).strip().lower()
        if source_type not in EXTERNAL_TYPES:
            raise ValueError(
                f"{path.name}: '{raw.get('archivo')}' tiene tipo {source_type!r}; "
                f"debe ser uno de {EXTERNAL_TYPES}."
            )
        info = SourceInfo(
            file=str(raw["archivo"]),
            source_type=source_type,
            topic=str(raw.get("tema", "")),
            title=str(raw.get("titulo") or raw["archivo"]),
            year=int(raw["anio"]) if raw.get("anio") else None,
        )
        catalog[_file_key(info.file)] = info
    return catalog


def source_for(pdf_path: Path, catalog: dict[str, SourceInfo]) -> SourceInfo:
    info = catalog.get(_file_key(Path(pdf_path).name))
    if info is None:
        raise ValueError(
            f"{Path(pdf_path).name} no esta en {CATALOG_FILENAME}. Agregalo con su "
            "tipo (guia o revision), tema y titulo antes de indexar."
        )
    return info


def source_label(source_type: str) -> str:
    return SOURCE_TYPE_LABELS.get(source_type, source_type)
