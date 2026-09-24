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
import re
import sys
import unicodedata
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

from gpc_rag.trees.registry import list_trees
from gpc_rag.trees.schema import DecisionTree

# Umbral de coincidencia de terminos para considerar una cita fiel. No es
# 100% porque el propio texto de redaccion de la cita (conectores, "Nota:",
# etc.) no tiene por que aparecer literal en el PDF -- lo que importa es que
# los terminos clinicos y numericos distintivos si esten.
UMBRAL_COINCIDENCIA = 0.6

# Un token cuenta como "termino clinico distintivo" si tiene al menos esta
# longitud (evita ruido de palabras cortas que no son stopwords pero tampoco
# aportan senal, como abreviaturas truncadas por la tokenizacion).
LONGITUD_MINIMA_TERMINO = 4

# La linea (1-indexed=2, 0-indexed=1) donde esta plantilla de revista imprime
# el numero de pagina -- ver docstring de _build_page_map.
LINEAS_MINIMAS_PARA_NUMERO_PAGINA = 2

_STOPWORDS = {
    "el", "la", "los", "las", "de", "del", "en", "y", "o", "a", "que", "con", "no",
    "por", "si", "un", "una", "al", "para", "nota", "tabla", "pag", "pagina",
}


# El PDF usa simbolos unicode (≤, ≥) y subindices (PaO₂, FiO₂) que no tienen
# equivalente ASCII directo -- sin esto, terminos que SI estan en el
# documento se reportarian como "no encontrados" solo por la codificacion,
# lo que seria un falso positivo del verificador, no un error real del
# arbol. Descubierto corriendo este mismo script contra el PDF real.
_REEMPLAZOS_SIMBOLOS = {
    "≤": "<=",
    "≥": ">=",
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
}


def _normalizar(texto: str) -> str:
    for simbolo, reemplazo in _REEMPLAZOS_SIMBOLOS.items():
        texto = texto.replace(simbolo, reemplazo)
    sin_acentos = "".join(
        c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn"
    )
    return sin_acentos.lower()


def _terminos_relevantes(quote: str) -> set[str]:
    """Extrae los terminos clinicos/numericos distintivos de una cita.

    Quita las anotaciones que agrega el propio script generador (parentesis
    del estilo "(Tabla 3, pag. 199)"), y luego tokeniza -- se quedan palabras
    de 4+ letras (terminos clinicos) y tokens numericos/porcentajes (los
    umbrales son lo mas facil de verificar y lo mas importante de no
    desalinear).
    """
    sin_anotaciones = re.sub(r"\([^()]*\)", "", quote)
    normalizado = _normalizar(sin_anotaciones)
    tokens = re.findall(r"[a-z0-9%/.<>=+-]+", normalizado)
    return {
        t
        for t in tokens
        if t not in _STOPWORDS
        and (len(t) >= LONGITUD_MINIMA_TERMINO or any(ch.isdigit() for ch in t))
    }


def _build_page_map(doc: fitz.Document) -> dict[int, int]:
    """Mapea numero de pagina IMPRESO (el que aparece en las citas) -> indice PDF (0-based).

    El PDF suele traer paginas de portada/indice sin numerar antes de que
    empiece la numeracion impresa del articulo, asi que el indice de pymupdf
    y el numero de pagina citado en las guias casi nunca coinciden. No basta
    con buscar "un numero solo en su propia linea" en cualquier parte de la
    pagina: las tablas clinicas (CURB-65, PSI, etc.) tienen celdas de una
    sola cifra que tambien matchean ese patron. En esta plantilla de revista
    el numero de pagina esta siempre en la segunda linea del texto extraido
    (justo despues del encabezado de autor/titulo), asi que se busca solo
    ahi -- si el layout de otra guia fuera distinto, esta funcion es el
    unico lugar que hay que ajustar.
    """
    mapa: dict[int, int] = {}
    for idx in range(len(doc)):
        lineas = doc[idx].get_text().strip().splitlines()
        if len(lineas) < LINEAS_MINIMAS_PARA_NUMERO_PAGINA:
            continue
        segunda_linea = lineas[1].strip()
        if re.fullmatch(r"\d{1,4}", segunda_linea):
            mapa[int(segunda_linea)] = idx
    return mapa


def _find_pdf(pdf_dir: Path, gpc_source: str) -> Path | None:
    objetivo = _normalizar(gpc_source)
    for candidato in pdf_dir.rglob("*.pdf"):
        if _normalizar(candidato.name) == objetivo or _normalizar(candidato.stem) in objetivo:
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

        texto_pagina = _normalizar(doc[pdf_idx].get_text())
        terminos = _terminos_relevantes(quote)
        if not terminos:
            continue
        encontrados = {t for t in terminos if t in texto_pagina}
        cobertura = len(encontrados) / len(terminos)

        if cobertura < UMBRAL_COINCIDENCIA:
            faltantes = sorted(terminos - encontrados)
            problemas.append(
                f"  [{node.node_id}] pag. {page}: cobertura {cobertura:.0%} "
                f"(umbral {UMBRAL_COINCIDENCIA:.0%}). Terminos no encontrados: {faltantes}"
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
        page_map = _build_page_map(doc)
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
