"""Tablas transcritas y revisadas de cada protocolo.

La extraccion automatica del PDF (pymupdf4llm) desordena las tablas de los
protocolos: parte filas, corta tablas entre paginas y no lee las que son
imagenes. Como esas tablas llevan farmacos y dosis, se transcriben a mano en
un archivo junto al PDF (`<nombre del pdf>.tablas.yaml`) y una persona las
revisa contra el PDF antes de indexar (`revisado: true`).

Cada fila se indexa como un fragmento autocontenido: titulo de la tabla,
cada columna con su valor y las notas al pie que le aplican. Asi el modelo
no tiene que reconstruir a que fila pertenece una nota o un valor suelto.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from gpc_rag.common.models import Chunk

SUFFIX = ".tablas.yaml"


@dataclass
class TableRow:
    cells: dict[str, str]
    notes: list[str] = field(default_factory=list)


@dataclass
class CuratedTable:
    id: str
    title: str
    page: int
    rows: list[TableRow]
    general_notes: list[str] = field(default_factory=list)


@dataclass
class CuratedTables:
    document: str
    tables: list[CuratedTable]
    drop_extracted_tables_on_pages: set[int]


def tables_path_for(pdf_path: Path) -> Path:
    return pdf_path.with_name(pdf_path.stem + SUFFIX)


def load_curated_tables(pdf_path: Path) -> CuratedTables | None:
    """Tablas revisadas del PDF, o None si no hay archivo de tablas.

    Falla si el archivo existe pero no esta marcado como revisado: no se
    indexan dosis transcritas que nadie comparo con el PDF.
    """
    path = tables_path_for(Path(pdf_path))
    if not path.is_file():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data.get("documento") != Path(pdf_path).name:
        raise ValueError(
            f"{path.name} declara el documento {data.get('documento')!r}, no {Path(pdf_path).name!r}."
        )
    if data.get("revisado") is not True:
        raise ValueError(
            f"{path.name} no esta marcado como revisado. Comparalo con el PDF, "
            "revisa 'pendiente_revision' y cambia 'revisado' a true antes de indexar."
        )
    tables = []
    for raw in data.get("tablas") or []:
        rows = [
            TableRow(
                cells={str(k): str(v) for k, v in (row.get("celdas") or {}).items()},
                notes=[str(n) for n in row.get("notas") or []],
            )
            for row in raw.get("filas") or []
        ]
        if not rows:
            raise ValueError(f"{path.name}: la tabla {raw.get('id')!r} no tiene filas.")
        tables.append(
            CuratedTable(
                id=str(raw["id"]),
                title=str(raw["titulo"]),
                page=int(raw["pagina"]),
                rows=rows,
                general_notes=[str(n) for n in raw.get("notas_generales") or []],
            )
        )
    return CuratedTables(
        document=str(data["documento"]),
        tables=tables,
        drop_extracted_tables_on_pages={
            int(p) for p in data.get("quitar_tablas_extraidas_de_paginas") or []
        },
    )


def row_text(table: CuratedTable, row: TableRow) -> str:
    """Texto autocontenido de una fila: titulo, columnas y notas."""
    lines = [f"{table.title}."]
    lines += [
        f"{column}: {value}" for column, value in row.cells.items() if value.strip()
    ]
    notes = row.notes + table.general_notes
    if notes:
        lines.append("Notas de la tabla para esta fila:")
        lines += [f"- {note}" for note in notes]
    return "\n".join(lines)


def table_chunks(curated: CuratedTables, start_index: int = 0) -> list[Chunk]:
    chunks = []
    index = start_index
    for table in curated.tables:
        for row in table.rows:
            chunks.append(
                Chunk(
                    text=row_text(table, row),
                    source_file=curated.document,
                    section=table.title,
                    page=table.page,
                    chunk_index=index,
                    chunk_id=str(uuid.uuid4()),
                )
            )
            index += 1
    return chunks


_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")


def strip_markdown_tables(text: str) -> str:
    """Quita las tablas markdown (lineas que empiezan y terminan con '|')."""
    kept = [line for line in text.splitlines() if not _TABLE_LINE_RE.match(line)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
