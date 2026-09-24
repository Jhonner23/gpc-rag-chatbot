"""Extraccion automatica de tablas de criterios desde un PDF de GPC -- sin LLM.

Pipeline determinista (mismo input -> siempre el mismo output, nada
probabilistico): usa `pdfplumber` para detectar tablas por su geometria
real en el PDF (lineas, bordes, espaciado), no por interpretar el
contenido, y luego parsea filas con reglas fijas de columnas/encabezados
para los dos patrones de instrumento clinico que ya soporta el bosque (ver
`trees/builders.py`):

- Tabla "suma de puntaje" (CURB-65/CRB-65, Tabla 3 pag. 199 de la guia
  NAC 2026): encabezado + una fila por criterio, con una columna de "1"/"-"
  por variante del instrumento (CURB-65 vs CRB-65).
- Tabla "mayor o N menores" (criterios IDSA/ATS, Tabla 5 pag. 199): dos
  celdas de texto libre ("Criterios mayores (>=1 ...)" / "Criterios menores
  (>=N ...)") cada una seguida de una celda con los criterios en vinetas.

Decision explicita del proyecto: cuando una tabla NO matchea con confianza
uno de estos dos patrones (encabezado desconocido, puntajes ponderados en
vez de +1 por criterio, formato de rango no reconocido), el pipeline
levanta `TablaNoReconocidaError` con el motivo exacto en vez de adivinar o
producir un arbol parcial -- un falso "exito" silencioso es peor que un
fallo explicito que un humano completa a mano (como se hizo con los dos
arboles actuales antes de que este modulo existiera).

Lo que este modulo SI decide por si solo (mecanico, sin ambiguedad): que
criterios hay, sus valores/umbrales, y la cita textual de cada uno (se arma
directamente de las celdas de la tabla, no se inventa). Lo que NO decide:
la redaccion final de la pregunta que le hace el wizard al usuario -- genera
una version literal ("Cumple: <texto de la tabla>?") a partir del texto de
la celda, pensada para que un humano la revise y la pula antes de comitear
el arbol, igual que ya se revisa cualquier otro `source_quote` con
`trees/fidelity.py`.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import fitz  # pymupdf
import pdfplumber

from gpc_rag.trees.builders import Criterio
from gpc_rag.trees.fidelity import build_page_map

Fila = list[str | None]

# Una tabla reconocible necesita, como minimo, una fila de encabezado y una
# fila de datos.
MIN_FILAS_CON_ENCABEZADO = 2


class TablaNoReconocidaError(ValueError):
    """Una tabla no matchea, con confianza, ninguno de los patrones soportados."""


def _texto(celda: str | None) -> str:
    """Normaliza una celda de pdfplumber: colapsa saltos de linea internos a espacios."""
    return re.sub(r"\s+", " ", (celda or "")).strip()


def _slug(texto: str) -> str:
    """Nombre de variable a partir del texto de un criterio (ej. 'BUN >=20 mg/dL' -> 'bun_20_mg_dl').

    Translitera acentos (Confusión -> confusion) en vez de simplemente
    descartar los caracteres no-ASCII -- lo segundo dejaba nombres de
    variable confusos como 'confusi_n'.
    """
    sin_acentos = "".join(
        c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn"
    )
    normalizado = re.sub(r"[^a-zA-Z0-9]+", "_", sin_acentos).strip("_").lower()
    return normalizado or "criterio"


def extraer_tablas_de_pagina(pdf_path: Path, pagina_impresa: int) -> list[list[Fila]]:
    """Devuelve las tablas detectadas en la pagina IMPRESA `pagina_impresa` (no el indice del PDF).

    Reusa el mismo `build_page_map` que ya usa `trees/fidelity.py` para
    traducir numero de pagina impreso -> indice real del PDF, asi que el
    mapeo es consistente en todo el proyecto.
    """
    with fitz.open(pdf_path) as doc_fitz:
        page_map = build_page_map(doc_fitz)

    pdf_idx = page_map.get(pagina_impresa)
    if pdf_idx is None:
        raise TablaNoReconocidaError(
            f"no se pudo ubicar la pagina impresa {pagina_impresa} en {pdf_path.name} "
            "(ver trees/fidelity.py:build_page_map si el layout de esta guia es distinto)"
        )

    with pdfplumber.open(pdf_path) as doc_plumber:
        tablas: list[list[Fila]] = doc_plumber.pages[pdf_idx].extract_tables()
    return tablas


_COLUMNA_CRITERIO = 0  # primera columna: nombre del criterio, siempre


def parsear_criterios_suma_puntaje(
    filas: list[Fila], *, columna_objetivo: str, pagina: int, referencia_tabla: str
) -> list[Criterio]:
    """Parsea una tabla tipo CURB-65: encabezado + una fila por criterio.

    `columna_objetivo` es el nombre de columna (ej. "CURB-65") cuya
    variante del instrumento se quiere construir -- una fila cuenta como
    criterio de esa variante solo si su celda en esa columna es "1"; si es
    "-"/vacia, el criterio no aplica a esta variante y se omite (asi es
    como CRB-65 excluye BUN). Cualquier otro valor (puntaje ponderado tipo
    "+20") hace fallar la extraccion explicitamente: este builder solo
    soporta instrumentos donde cada criterio suma exactamente 1 punto.
    """
    if not filas or len(filas) < MIN_FILAS_CON_ENCABEZADO:
        raise TablaNoReconocidaError("la tabla no tiene encabezado + al menos una fila de criterio")

    encabezado = [_texto(c) for c in filas[0]]
    if columna_objetivo not in encabezado:
        raise TablaNoReconocidaError(
            f"no se encontro la columna {columna_objetivo!r} en el encabezado {encabezado!r} "
            "-- ¿es realmente una tabla de suma de puntaje?"
        )
    idx_columna = encabezado.index(columna_objetivo)

    criterios: list[Criterio] = []
    for fila in filas[1:]:
        if fila[_COLUMNA_CRITERIO] is None:
            raise TablaNoReconocidaError(
                f"fila {fila!r} no trae nombre de criterio en la primera columna -- esto suele "
                "pasar en tablas con sub-categorias (ej. Fine/PSI), que este parser no soporta"
            )
        if idx_columna >= len(fila):
            raise TablaNoReconocidaError(f"fila {fila!r} no tiene columna {idx_columna} ({columna_objetivo!r})")

        nombre = _texto(fila[_COLUMNA_CRITERIO])
        valor = _texto(fila[idx_columna])
        if valor in ("-", "", "0"):
            continue
        if valor != "1":
            raise TablaNoReconocidaError(
                f"criterio {nombre!r}: valor {valor!r} en columna {columna_objetivo!r} no es '1' "
                "-- este builder solo soporta instrumentos con puntaje +1 por criterio (no "
                "puntajes ponderados), ver construir_arbol_suma_puntaje en trees/builders.py"
            )

        columnas_extra = " | ".join(
            f"{encabezado[i]}: {_texto(fila[i])}" for i in range(1, len(fila)) if i != idx_columna
        )
        cita = f"{nombre} | {columna_objetivo}: {valor}" + (f" | {columnas_extra}" if columnas_extra else "")
        criterios.append(
            Criterio(
                variable=_slug(nombre),
                pregunta=f"Cumple: {nombre}?",
                source_quote=f"{cita} ({referencia_tabla}, pag. {pagina}).",
            )
        )

    if not criterios:
        raise TablaNoReconocidaError(f"ningun criterio aplica a la columna {columna_objetivo!r}")
    return criterios


_RANGO_UNICO = re.compile(r"^(\d+)$")
_RANGO_ENTRE = re.compile(r"^(\d+)\s*-\s*(\d+)$")
_RANGO_ABIERTO_ARRIBA = re.compile(r"^[≥>]=?\s*(\d+)$")
_RANGO_ABIERTO_ABAJO = re.compile(r"^[≤<]=?\s*(\d+)$")


def _expandir_rango(texto: str, puntaje_maximo: int) -> list[int]:
    """Convierte '0 - 1' / '2' / '>=3' / '<=70' a la lista de puntajes enteros que cubre."""
    t = texto.strip()
    if m := _RANGO_UNICO.match(t):
        return [int(m.group(1))]
    if m := _RANGO_ENTRE.match(t):
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    if m := _RANGO_ABIERTO_ARRIBA.match(t):
        return list(range(int(m.group(1)), puntaje_maximo + 1))
    if m := _RANGO_ABIERTO_ABAJO.match(t):
        return list(range(0, int(m.group(1)) + 1))
    raise TablaNoReconocidaError(f"formato de rango de puntaje no reconocido: {texto!r}")


def parsear_interpretacion_suma_puntaje(  # noqa: PLR0913 -- keyword-only, ver nota en trees/builders.py
    filas: list[Fila],
    *,
    n_criterios: int,
    columna_rango: str,
    columna_recomendacion: list[str],
    pagina: int,
    referencia_tabla: str,
) -> dict[int, tuple[str, str]]:
    """Parsea la tabla de interpretacion (rango de puntaje -> riesgo/manejo).

    `columna_recomendacion` puede ser mas de una columna (ej. "Riesgo de
    mortalidad" + "Manejo sugerido") -- se concatenan para el texto de la
    recomendacion. Devuelve un dict puntaje -> (recomendacion, source_quote),
    con una entrada por cada puntaje 0..n_criterios (expandiendo rangos como
    "0 - 1" o ">=3") -- listo para pasarle directo a
    `construir_arbol_suma_puntaje`.
    """
    if not filas or len(filas) < MIN_FILAS_CON_ENCABEZADO:
        raise TablaNoReconocidaError("la tabla de interpretacion no tiene encabezado + filas")

    encabezado = [_texto(c) for c in filas[0]]
    if columna_rango not in encabezado:
        raise TablaNoReconocidaError(f"no se encontro la columna {columna_rango!r} en {encabezado!r}")
    for col in columna_recomendacion:
        if col not in encabezado:
            raise TablaNoReconocidaError(f"no se encontro la columna {col!r} en {encabezado!r}")

    idx_rango = encabezado.index(columna_rango)
    idxs_recomendacion = [encabezado.index(c) for c in columna_recomendacion]

    interpretacion: dict[int, tuple[str, str]] = {}
    for fila in filas[1:]:
        rango_texto = _texto(fila[idx_rango])
        puntajes = _expandir_rango(rango_texto, n_criterios)
        partes_recomendacion = [_texto(fila[i]) for i in idxs_recomendacion]
        recomendacion = ", ".join(f"{encabezado[i]} {v}" for i, v in zip(idxs_recomendacion, partes_recomendacion, strict=True))
        cita = (
            f"Puntaje {rango_texto}: " + ", ".join(partes_recomendacion)
            + f" ({referencia_tabla}, pag. {pagina})."
        )
        for puntaje in puntajes:
            if puntaje > n_criterios:
                continue
            interpretacion[puntaje] = (recomendacion, cita)

    faltantes = set(range(n_criterios + 1)) - set(interpretacion)
    if faltantes:
        raise TablaNoReconocidaError(
            f"la tabla de interpretacion no cubre los puntajes {sorted(faltantes)} "
            f"(instrumento de {n_criterios} criterios, puntajes posibles 0..{n_criterios})"
        )
    return interpretacion


_UMBRAL_MENORES = re.compile(r"[≥>]=?\s*(\d+)")


def parsear_mayor_o_menores(
    filas: list[Fila],
    *,
    etiqueta_mayores: str = "mayores",
    etiqueta_menores: str = "menores",
    pagina: int,
    referencia_tabla: str,
) -> tuple[list[Criterio], list[Criterio], int]:
    """Parsea una tabla tipo IDSA/ATS: dos bloques de texto (mayores / menores),
    cada uno con una fila de encabezado ('Criterios mayores (>=1 ...)') seguida
    de una fila cuya unica celda trae los criterios separados por vinetas ('•').

    Devuelve (criterios_mayores, criterios_menores, umbral_menores) -- listo
    para pasarle a `construir_arbol_mayor_o_n_menores`.
    """
    texto_filas = [_texto(fila[0]) if fila else "" for fila in filas]

    idx_mayores = next((i for i, t in enumerate(texto_filas) if etiqueta_mayores in t.lower()), None)
    idx_menores = next((i for i, t in enumerate(texto_filas) if etiqueta_menores in t.lower()), None)
    if idx_mayores is None or idx_menores is None:
        raise TablaNoReconocidaError(
            f"no se encontraron las etiquetas {etiqueta_mayores!r}/{etiqueta_menores!r} "
            f"en las filas de la tabla: {texto_filas!r}"
        )
    if idx_mayores + 1 >= len(filas) or idx_menores + 1 >= len(filas):
        raise TablaNoReconocidaError("falta la fila de vinetas justo despues del encabezado de mayores/menores")

    def _vinetas(texto: str) -> list[str]:
        partes = [p.strip() for p in texto.split("•") if p.strip()]
        if not partes:
            raise TablaNoReconocidaError(f"no se encontraron vinetas ('•') en: {texto!r}")
        return partes

    m = _UMBRAL_MENORES.search(texto_filas[idx_menores])
    if not m:
        raise TablaNoReconocidaError(
            f"no se pudo leer el umbral numerico (ej. '>=3') en: {texto_filas[idx_menores]!r}"
        )
    umbral_menores = int(m.group(1))

    def _a_criterios(texto_vinetas: str) -> list[Criterio]:
        return [
            Criterio(
                variable=_slug(v),
                pregunta=f"Cumple: {v}?",
                source_quote=f"{v} ({referencia_tabla}, pag. {pagina}).",
            )
            for v in _vinetas(texto_vinetas)
        ]

    criterios_mayores = _a_criterios(texto_filas[idx_mayores + 1])
    criterios_menores = _a_criterios(texto_filas[idx_menores + 1])
    return criterios_mayores, criterios_menores, umbral_menores
