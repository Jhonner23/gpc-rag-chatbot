"""CLI: extrae un arbol de decision desde una tabla de la GPC -- sin LLM.

Junta `trees/table_extraction.py` (deteccion/parseo de tablas por geometria
+ regex, determinista) con `trees/builders.py` (enumeracion de nodos) para
que el flujo completo -- PDF -> tablas -> criterios -> arbol JSON -- sea un
solo comando reproducible, en vez de los scripts `python3 -c ...` ad-hoc
usados para validar el pipeline durante el desarrollo.

Dos subcomandos:

    listar      Muestra las tablas detectadas en una pagina (indice,
                dimensiones, encabezado) -- para ubicar el indice de tabla
                correcto antes de escribir la receta.
    construir   Lee una "receta" JSON (que tabla(s), que columnas, que
                textos de hoja) y escribe el arbol resultante en
                trees/data/<gpc_slug>/<tree_id>.json, despues de validarlo
                contra el schema Y contra la fidelidad de sus citas (misma
                verificacion que scripts/verify_tree_citations.py).

La receta es lo unico que un humano escribe a mano: que columna de la tabla
usar, y el texto de las 3 hojas fijas de un instrumento "mayor o N menores"
(redaccion clinica, no estructura -- eso lo sigue decidiendo una persona,
igual que ya se hacia con los generadores gen_*_tree.py). Todo lo demas
--que criterios hay, sus citas, el armado de nodos-- es mecanico y
reproducible: la misma receta sobre el mismo PDF siempre da el mismo JSON.

Ejemplos:

    python scripts/extract_tree_from_table.py listar \\
        --pdf "data/01_raw/gpc/Guia de practica clinica Colombiana NAC 2026.pdf" \\
        --pagina 199

    python scripts/extract_tree_from_table.py construir \\
        --pdf "data/01_raw/gpc/Guia de practica clinica Colombiana NAC 2026.pdf" \\
        --receta recetas/curb65.json \\
        --out src/gpc_rag/trees/data/nac-2026/severidad-hospitalizacion-v2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import fitz  # pymupdf

from gpc_rag.trees.builders import (
    Fuente,
    construir_arbol_mayor_o_n_menores,
    construir_arbol_suma_puntaje,
)
from gpc_rag.trees.fidelity import UMBRAL_COINCIDENCIA, build_page_map, cobertura_cita, normalizar
from gpc_rag.trees.schema import DecisionTree
from gpc_rag.trees.table_extraction import (
    TablaNoReconocidaError,
    extraer_tablas_de_pagina,
    parsear_criterios_suma_puntaje,
    parsear_interpretacion_suma_puntaje,
    parsear_mayor_o_menores,
)

_TIPOS_SOPORTADOS = ("suma_puntaje", "mayor_o_menores")


def _cmd_listar(args: argparse.Namespace) -> int:
    tablas = extraer_tablas_de_pagina(args.pdf, args.pagina)
    if not tablas:
        print(f"No se detecto ninguna tabla en la pagina impresa {args.pagina} de {args.pdf.name}.")
        return 0

    print(f"{len(tablas)} tabla(s) detectada(s) en la pagina impresa {args.pagina} de {args.pdf.name}:\n")
    for idx, tabla in enumerate(tablas):
        n_filas = len(tabla)
        n_cols = max((len(f) for f in tabla), default=0)
        encabezado = tabla[0] if tabla else []
        print(f"  Tabla {idx}: {n_filas} filas x {n_cols} columnas")
        print(f"    encabezado/primera fila: {encabezado!r}")
    print("\nUsa el indice de la tabla que corresponda en tu receta (campo 'tabla'/'tabla_criterios'/...).")
    return 0


def _hoja_desde_receta(hoja: dict[str, Any], fuente: Fuente) -> tuple[str, str, Fuente, str, str | None]:
    return (
        hoja["node_id"],
        hoja["recomendacion"],
        fuente,
        hoja["source_quote"],
        hoja.get("evidence_level"),
    )


def _construir_suma_puntaje(receta: dict[str, Any], tablas: list, args: argparse.Namespace) -> dict[str, Any]:
    fuente_criterios = Fuente(
        guide_file=receta["gpc_source"], section=receta["seccion_criterios"], page=args.pagina
    )
    fuente_interpretacion = Fuente(
        guide_file=receta["gpc_source"], section=receta["seccion_interpretacion"], page=args.pagina
    )

    criterios = parsear_criterios_suma_puntaje(
        tablas[receta["tabla_criterios"]],
        columna_objetivo=receta["columna_objetivo"],
        pagina=args.pagina,
        referencia_tabla=receta["seccion_criterios"],
    )
    interpretacion = parsear_interpretacion_suma_puntaje(
        tablas[receta["tabla_interpretacion"]],
        n_criterios=len(criterios),
        columna_rango=receta["columna_rango"],
        columna_recomendacion=receta["columna_recomendacion"],
        pagina=args.pagina,
        referencia_tabla=receta["seccion_interpretacion"],
    )
    arbol_dict: dict[str, Any] = construir_arbol_suma_puntaje(
        tree_id=receta["tree_id"],
        title=receta["title"],
        description=receta["description"],
        gpc_source=receta["gpc_source"],
        criterios=criterios,
        fuente_criterios=fuente_criterios,
        fuente_interpretacion=fuente_interpretacion,
        interpretacion=interpretacion,
    )
    return arbol_dict


def _construir_mayor_o_menores(receta: dict[str, Any], tablas: list, args: argparse.Namespace) -> dict[str, Any]:
    fuente = Fuente(guide_file=receta["gpc_source"], section=receta["seccion"], page=args.pagina)

    mayores, menores, umbral = parsear_mayor_o_menores(
        tablas[receta["tabla"]], pagina=args.pagina, referencia_tabla=receta["seccion"]
    )
    arbol_dict: dict[str, Any] = construir_arbol_mayor_o_n_menores(
        tree_id=receta["tree_id"],
        title=receta["title"],
        description=receta["description"],
        gpc_source=receta["gpc_source"],
        criterios_mayores=mayores,
        criterios_menores=menores,
        umbral_menores=umbral,
        fuente_criterios=fuente,
        hoja_directa=_hoja_desde_receta(receta["hoja_directa"], fuente),
        hoja_umbral_menores=_hoja_desde_receta(receta["hoja_umbral_menores"], fuente),
        hoja_sin_indicacion=_hoja_desde_receta(receta["hoja_sin_indicacion"], fuente),
    )
    return arbol_dict


def _verificar_fidelidad(tree: DecisionTree, doc: fitz.Document, page_map: dict[int, int]) -> list[str]:
    """Misma verificacion que scripts/verify_tree_citations.py, corrida sobre el arbol recien construido.

    No tiene sentido escribir un arbol cuyas propias citas no pasarian la
    verificacion de fidelidad -- se corre aqui mismo, antes de escribir el
    JSON, en vez de dejarlo para un paso separado que alguien podria saltarse.
    """
    problemas: list[str] = []
    for node in tree.nodes.values():
        if node.source.page is None or not node.source_quote:
            continue
        pdf_idx = page_map.get(node.source.page)
        if pdf_idx is None:
            problemas.append(f"  [{node.node_id}] no se encontro la pagina impresa {node.source.page} en el PDF")
            continue
        resultado = cobertura_cita(node.source_quote, normalizar(doc[pdf_idx].get_text()))
        if not resultado.es_fiel:
            problemas.append(
                f"  [{node.node_id}] pag. {node.source.page}: cobertura {resultado.cobertura:.0%} "
                f"(umbral {UMBRAL_COINCIDENCIA:.0%}). Terminos no encontrados: {sorted(resultado.terminos_faltantes)}"
            )
    return problemas


def _cmd_construir(args: argparse.Namespace) -> int:
    receta = json.loads(args.receta.read_text(encoding="utf-8"))
    tipo = receta.get("tipo")
    if tipo not in _TIPOS_SOPORTADOS:
        print(f"receta['tipo'] debe ser uno de {_TIPOS_SOPORTADOS}, no {tipo!r}", file=sys.stderr)
        return 2

    try:
        tablas = extraer_tablas_de_pagina(args.pdf, args.pagina)
        if tipo == "suma_puntaje":
            arbol_dict = _construir_suma_puntaje(receta, tablas, args)
        else:
            arbol_dict = _construir_mayor_o_menores(receta, tablas, args)
    except TablaNoReconocidaError as exc:
        print(f"Extraccion fallida (no se adivina, se avisa): {exc}", file=sys.stderr)
        return 1
    except (KeyError, IndexError) as exc:
        print(f"La receta no trae un campo esperado o el indice de tabla no existe: {exc}", file=sys.stderr)
        return 2

    tree = DecisionTree.model_validate(arbol_dict)
    print(f"Arbol construido: {tree.tree_id} ({len(tree.nodes)} nodos).")

    with fitz.open(args.pdf) as doc:
        page_map = build_page_map(doc)
        problemas = _verificar_fidelidad(tree, doc, page_map)

    if problemas:
        print(f"\n{len(problemas)} nodo(s) no pasan la verificacion de fidelidad -- no se escribe el archivo:")
        for p in problemas:
            print(p, file=sys.stderr)
        return 1
    print("Fidelidad de citas: OK (todas superan el umbral).")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(arbol_dict, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nescrito: {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="comando", required=True)

    p_listar = sub.add_parser("listar", help="Lista las tablas detectadas en una pagina del PDF.")
    p_listar.add_argument("--pdf", type=Path, required=True)
    p_listar.add_argument("--pagina", type=int, required=True, help="numero de pagina IMPRESO (no el indice del PDF)")
    p_listar.set_defaults(func=_cmd_listar)

    p_construir = sub.add_parser("construir", help="Construye un arbol a partir de una receta JSON.")
    p_construir.add_argument("--pdf", type=Path, required=True)
    p_construir.add_argument("--pagina", type=int, required=True, help="numero de pagina IMPRESO")
    p_construir.add_argument("--receta", type=Path, required=True, help="archivo JSON con la receta (ver docstring)")
    p_construir.add_argument("--out", type=Path, required=True, help="ruta de salida del arbol JSON")
    p_construir.set_defaults(func=_cmd_construir)

    args = parser.parse_args()
    if not args.pdf.is_file():
        print(f"No existe el PDF: {args.pdf}", file=sys.stderr)
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
