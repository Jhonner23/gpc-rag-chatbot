"""Verifica que las citas de los arboles de decision sean fieles al PDF fuente.

Cierra el requisito de replicabilidad: alguien que clone el repo y tenga su
propia copia de las GPC (el PDF no esta versionado en git -- ver .gitignore)
puede correr este script y confirmar, sin depender de esta conversacion ni
de que "Claude dijo que estaba bien", que cada `source_quote` de cada nodo
del bosque (trees/data/**/*.json) sigue correspondiendo a lo que dice
realmente la pagina citada del documento.

No es una comparacion de substring exacto: los `source_quote` del bosque son
resumenes compactos de tablas (ej. "Confusion | CURB-65: 1 | CRB-65: 1"), no
texto copiado literal -- extraer una tabla de un PDF reordena filas/columnas,
asi que un string identico no existe en el documento. En su lugar, se extrae
el texto real de la pagina citada y se verifica que los terminos y valores
clinicamente relevantes de la cita (numeros, umbrales, terminos medicos)
aparezcan ahi. Es exactamente el patron de "fidelidad de extraccion, no
re-validacion clinica" descrito en la seccion 10.1 de la arquitectura: no se
re-evalua si el criterio es correcto clinicamente (eso ya lo aprobo la
clinica al proveer el documento) -- solo si el arbol representa fielmente lo
que el documento dice.

Uso:
    python scripts/verify_tree_citations.py --pdf-dir data/01_raw/gpc

Busca, dentro de --pdf-dir (recursivo), un archivo cuyo nombre coincida con
el `gpc_source` de cada arbol (ej. "Guia de practica clinica Colombiana NAC
2026.pdf"). Si no lo encuentra, avisa y sigue con los demas arboles -- no es
necesario tener todas las guias para verificar las que si se tengan.

Requiere pymupdf (`uv add pymupdf` o `pip install pymupdf`), no forma parte
de las dependencias de produccion (esto es una herramienta offline, igual
que los scripts gen_*_tree.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import fitz  # pymupdf
except ImportError as exc:
    print(
        "Falta pymupdf. Instalalo con: uv add --group dev pymupdf  (o pip install pymupdf)",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from gpc_rag.trees.fidelity import (
    UMBRAL_COINCIDENCIA,
    build_page_map,
    cobertura_cita,
    normalizar,
)
from gpc_rag.trees.registry import list_trees
from gpc_rag.trees.schema import DecisionTree


def _find_pdf(pdf_dir: Path, gpc_source: str) -> Path | None:
    objetivo = normalizar(gpc_source)
    for candidato in pdf_dir.rglob("*.pdf"):
        if normalizar(candidato.name) == objetivo or normalizar(candidato.stem) in objetivo:
            return candidato
    return None


def _verificar_arbol(tree: DecisionTree, doc: fitz.Document, page_map: dict[int, int]) -> list[str]:
    problemas: list[str] = []
    for node in tree.nodes.values():
        page = node.source.page
        quote = node.source_quote
        if page is None or not quote:
            continue

        pdf_idx = page_map.get(page)
        if pdf_idx is None:
            problemas.append(
                f"  [{node.node_id}] no se encontro la pagina impresa {page} en el PDF "
                "(revisar manualmente)"
            )
            continue

        texto_pagina = normalizar(doc[pdf_idx].get_text())
        resultado = cobertura_cita(quote, texto_pagina)

        if not resultado.es_fiel:
            problemas.append(
                f"  [{node.node_id}] pag. {page}: cobertura {resultado.cobertura:.0%} "
                f"(umbral {UMBRAL_COINCIDENCIA:.0%}). "
                f"Terminos no encontrados: {sorted(resultado.terminos_faltantes)}"
            )
    return problemas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        required=True,
        help="Carpeta donde estan (tu copia de) las GPC en PDF -- se busca recursivamente.",
    )
    args = parser.parse_args()

    if not args.pdf_dir.is_dir():
        print(f"No existe la carpeta {args.pdf_dir}", file=sys.stderr)
        return 2

    entries = list_trees()
    if not entries:
        print("No hay arboles en el bosque (trees/data/ vacio).")
        return 0

    guias_por_fuente: dict[str, list] = {}
    for entry in entries:
        guias_por_fuente.setdefault(entry.tree.gpc_source, []).append(entry)

    hubo_problemas = False
    for gpc_source, arboles in guias_por_fuente.items():
        pdf_path = _find_pdf(args.pdf_dir, gpc_source)
        print(f"\n=== Guia: {gpc_source} ===")
        if pdf_path is None:
            print(f"  No se encontro el PDF en {args.pdf_dir} -- se omite (no es un error fatal).")
            continue
        print(f"  PDF: {pdf_path}")

        doc = fitz.open(pdf_path)
        page_map = build_page_map(doc)
        print(f"  Paginas mapeadas: {len(page_map)} de {len(doc)}")

        for entry in arboles:
            print(f"\n  --- Arbol: {entry.tree_id} ({len(entry.tree.nodes)} nodos) ---")
            problemas = _verificar_arbol(entry.tree, doc, page_map)
            if not problemas:
                print("  OK: todas las citas superan el umbral de fidelidad.")
            else:
                hubo_problemas = True
                print(f"  {len(problemas)} nodo(s) para revisar:")
                for p in problemas:
                    print(p)

    print()
    if hubo_problemas:
        print("Resultado: hay nodos que no alcanzan el umbral de fidelidad -- revisar antes de confiar en el arbol.")
        return 1
    print("Resultado: todas las citas verificadas son fieles al documento fuente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
